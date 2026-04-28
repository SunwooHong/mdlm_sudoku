from __future__ import annotations

import copy
import os
from typing import Iterable, List, Sequence, Tuple

import fsspec
import hydra
import lightning as L
import omegaconf
import torch
import torch.nn.functional as F

import dataloader
import diffusion
import utils
from oscar import sudoku_orbit as orbit


class OSCARDiffusion(diffusion.Diffusion):
    """OSCAR-lite for Sudoku.

    Directly targets logical equivariance by minimizing cross-transform score
    variance on the same logical state, with an anchor-aware task loss and a
    proximal KL term to a frozen reference model.
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        if not hasattr(config, "oscar"):
            raise ValueError("config.oscar is required for OSCAR training")
        self.oscar_cfg = config.oscar
        self.ref_backbone = copy.deepcopy(self.backbone)
        for p in self.ref_backbone.parameters():
            p.requires_grad_(False)
        self.ref_backbone.eval()
        self._t_grid = [float(x) for x in list(self.oscar_cfg.t_grid)]

    @torch.no_grad()
    def reset_reference(self) -> None:
        self.ref_backbone.load_state_dict(copy.deepcopy(self.backbone.state_dict()))
        self.ref_backbone.eval()
        for p in self.ref_backbone.parameters():
            p.requires_grad_(False)

    def _select_transforms(self) -> List[str]:
        names = list(orbit.TRANSFORM_NAMES)
        n = int(getattr(self.oscar_cfg, "num_transforms", len(names)))
        n = max(1, min(n, len(names)))
        always_include_identity = bool(getattr(self.oscar_cfg, "always_include_identity", True))
        if n >= len(names):
            return names
        if always_include_identity:
            rest = names[1:]
            perm = torch.randperm(len(rest), device=self.device).tolist()
            chosen = [rest[i] for i in perm[: max(0, n - 1)]]
            return ["identity"] + chosen
        perm = torch.randperm(len(names), device=self.device).tolist()
        return [names[i] for i in perm[:n]]

    def _stack_views(
        self,
        x0: torch.Tensor,
        attention_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        names: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_views = orbit.stack_orbit_views(x0, names, dim=1, stack_dim=1)
        attn_views = orbit.stack_orbit_views(attention_mask, names, dim=1, stack_dim=1)
        anc_views = orbit.stack_orbit_views(anchor_mask.long(), names, dim=1, stack_dim=1).bool()
        return x_views, attn_views, anc_views

    def _prepare_t(self, batch_size: int, t_scalar: float, device: torch.device) -> torch.Tensor:
        t = torch.full((batch_size,), float(t_scalar), dtype=self.dtype, device=device)
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            t = t + (1.0 / self.T)
            t = t.clamp(max=1.0)
        return t

    def _conditioning_and_move(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.change_of_variables:
            unet_conditioning = t[:, None]
            f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))[:, None]
        else:
            sigma, _ = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])
        return unet_conditioning, move_chance

    def _forward_with_backbone(self, x: torch.Tensor, sigma: torch.Tensor, backbone: torch.nn.Module) -> torch.Tensor:
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = backbone(x, sigma)
            if self.parameterization == "subs":
                return self._subs_parameterization(logits=logits, xt=x)
            if self.parameterization == "sedd":
                return self._sedd_parameterization(logits=logits, xt=x, sigma=sigma)
            if self.parameterization == "d3pm":
                return self._d3pm_parameterization(logits=logits)
            return logits

    def _shared_orbit_corruption(
        self,
        x_views: torch.Tensor,
        anchor_views: torch.Tensor,
        names: Sequence[str],
        move_chance: torch.Tensor,
    ) -> torch.Tensor:
        # x_views: [B, M, 81], move_chance: [B, 1]
        B, M, L = x_views.shape
        base_u = torch.rand(B, L, device=x_views.device)
        u_views = orbit.stack_orbit_views(base_u, names, dim=1, stack_dim=1)
        move = (u_views < move_chance[:, None, :]) & (~anchor_views.bool())
        xt_views = torch.where(move, torch.full_like(x_views, self.mask_index), x_views)
        return xt_views

    def _completion_scores(
        self,
        x_views: torch.Tensor,
        target_views: torch.Tensor,
        anchor_views: torch.Tensor,
        names: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M, L = x_views.shape
        total_scores = x_views.new_zeros((B, M), dtype=torch.float32)
        total_kl = x_views.new_zeros((B, M), dtype=torch.float32)
        vicreg_loss = x_views.new_zeros((), dtype=torch.float32)
        reps = None

        for idx, t_scalar in enumerate(self._t_grid):
            t = self._prepare_t(B, t_scalar, x_views.device)
            cond, move_chance = self._conditioning_and_move(t)
            xt_views = self._shared_orbit_corruption(x_views, anchor_views, names, move_chance)

            flat_xt = xt_views.reshape(B * M, L)
            flat_x0 = x_views.reshape(B * M, L)
            flat_target = target_views.reshape(B * M, L).float()
            cond_rep = cond.repeat_interleave(M, dim=0)

            logp = self.forward(flat_xt, cond_rep).view(B, M, L, -1)
            token_logp = torch.gather(logp, -1, x_views.unsqueeze(-1)).squeeze(-1)
            denom = target_views.float().sum(-1).clamp_min(1.0)
            total_scores = total_scores + (token_logp * target_views.float()).sum(-1) / denom

            with torch.no_grad():
                ref_logp = self._forward_with_backbone(flat_xt, cond_rep, self.ref_backbone).view(B, M, L, -1)
            kl = F.kl_div(logp, ref_logp.exp(), reduction="none").sum(-1)
            total_kl = total_kl + (kl * target_views.float()).sum(-1) / denom

            if idx == 0 and float(getattr(self.oscar_cfg, "lambda_vicreg", 0.0)) > 0.0:
                canon_logits = []
                for m, name in enumerate(names):
                    inv = orbit.get_inv_perm(name).to(logp.device)
                    canon_logits.append(logp[:, m].index_select(1, inv))
                reps = torch.stack(canon_logits, dim=1)

        total_scores = total_scores / len(self._t_grid)
        total_kl = total_kl / len(self._t_grid)

        eq_loss = total_scores.var(dim=1, unbiased=False).mean()
        prox_loss = total_kl.mean()

        if reps is not None:
            base_target = target_views[:, 0].float()  # identity view is always included by default
            masked = reps * base_target[:, None, :, None]
            flat = masked.reshape(B, M, -1)
            mean_view = flat.mean(dim=1, keepdim=True)
            inv_loss = ((flat - mean_view) ** 2).mean()

            z = flat.reshape(B * M, -1)
            z = z - z.mean(dim=0, keepdim=True)
            std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
            gamma = float(getattr(self.oscar_cfg, "vicreg_gamma", 1.0))
            var_loss = F.relu(gamma - std).mean()

            if z.shape[0] > 1:
                cov = (z.T @ z) / (z.shape[0] - 1)
                off_diag = cov.flatten()[:-1].view(cov.shape[0] - 1, cov.shape[1] + 1)[:, 1:].flatten()
                cov_loss = (off_diag ** 2).mean()
            else:
                cov_loss = z.new_zeros(())

            vicreg_loss = inv_loss
            vicreg_loss = vicreg_loss + float(getattr(self.oscar_cfg, "vicreg_var_weight", 25.0)) * var_loss
            vicreg_loss = vicreg_loss + float(getattr(self.oscar_cfg, "vicreg_cov_weight", 1.0)) * cov_loss

        return eq_loss, prox_loss, vicreg_loss

    def training_step(self, batch, batch_idx):
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(batch["input_ids"], dtype=torch.long)
        anchor_mask = batch.get("anchor_mask")
        if anchor_mask is None:
            anchor_mask = torch.zeros_like(batch["input_ids"], dtype=torch.bool)

        names = self._select_transforms()
        x_views, target_views, anchor_views = self._stack_views(
            batch["input_ids"], attention_mask, anchor_mask, names
        )
        B, M, L = x_views.shape

        flat_x = x_views.reshape(B * M, L)
        flat_target = target_views.reshape(B * M, L)
        flat_anchor = anchor_views.reshape(B * M, L)

        losses = self._loss(flat_x, flat_target, anchor_mask=flat_anchor)
        self.train_metrics.update(losses.nlls, losses.token_mask)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, sync_dist=True)

        eq_loss, prox_loss, vicreg_loss = self._completion_scores(
            x_views=x_views,
            target_views=target_views,
            anchor_views=anchor_views,
            names=names,
        )

        lambda_eq = float(getattr(self.oscar_cfg, "lambda_eq", 1.0))
        lambda_prox = float(getattr(self.oscar_cfg, "lambda_prox", 0.1))
        lambda_vicreg = float(getattr(self.oscar_cfg, "lambda_vicreg", 0.0))

        total = losses.loss + lambda_eq * eq_loss + lambda_prox * prox_loss + lambda_vicreg * vicreg_loss

        self.log("trainer/loss", total.detach(), on_step=True, on_epoch=False, sync_dist=True)
        self.log("oscar/task", losses.loss.detach(), on_step=True, on_epoch=False, sync_dist=True)
        self.log("oscar/eq", eq_loss.detach(), on_step=True, on_epoch=False, sync_dist=True)
        self.log("oscar/prox", prox_loss.detach(), on_step=True, on_epoch=False, sync_dist=True)
        if lambda_vicreg > 0:
            self.log("oscar/vicreg", vicreg_loss.detach(), on_step=True, on_epoch=False, sync_dist=True)
        return total

    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx)


def _load_initial_weights(model: OSCARDiffusion, ckpt_path: str | None) -> None:
    if not ckpt_path:
        model.reset_reference()
        return
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ignored_missing = [k for k in missing if k.startswith("ref_backbone.")]
    real_missing = [k for k in missing if not k.startswith("ref_backbone.")]
    if real_missing or unexpected:
        print("[OSCAR] load_state_dict warnings")
        print("  missing:", real_missing)
        print("  ignored ref missing:", ignored_missing[:4], "..." if len(ignored_missing) > 4 else "")
        print("  unexpected:", unexpected)
    model.reset_reference()


def train(config, logger, tokenizer):
    logger.info("Starting OSCAR training.")

    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            **config.wandb,
        )

    ckpt_path = None
    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path

    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)

    model = OSCARDiffusion(config, tokenizer=valid_ds.tokenizer)
    init_ckpt = getattr(config.oscar, "init_checkpoint", "")
    if not ckpt_path:
        if not init_ckpt:
            init_ckpt = getattr(config.eval, "checkpoint_path", "")
        if init_ckpt:
            logger.info(f"Initializing OSCAR from checkpoint: {init_ckpt}")
            _load_initial_weights(model, init_ckpt)
        else:
            logger.info("No init checkpoint provided; OSCAR starts from current random init.")
            model.reset_reference()

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
