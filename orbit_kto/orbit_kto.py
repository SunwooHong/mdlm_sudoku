
"""Orbit-KTO trainer for the SunwooHong/mdlm_sudoku codebase.

This trainer keeps the existing Diffusion model as the policy and adds:
  1) conditioned Sudoku infill sampling over D4 transforms,
  2) orbit-aware KTO reweighting using mixed-orbit counterexample counts,
  3) semi-online refresh of orbit buffers.

Intended usage:
    - add a `mode=orbit_kto` branch in main.py that calls train(...)
    - provide an `orbit_kto` config block (see accompanying YAML snippet)
    - use `data=sudoku9-anchors` so batches include `anchor_mask`

This implementation is single-process / single-GPU oriented for simplicity.
"""

from __future__ import annotations

import copy
import itertools
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import hydra.utils
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import dataloader
import diffusion
import utils
from orbit_kto.sudoku_orbit import (
    choose_transforms,
    exact_match,
    make_orbit_tensors,
    orbit_counterexample_weights,
    orbit_statistics,
)


@dataclass
class OrbitRecord:
    candidate: torch.Tensor
    solution: torch.Tensor
    attention_mask: torch.Tensor
    anchor_mask: torch.Tensor
    label: torch.Tensor
    w_pos: torch.Tensor
    w_neg: torch.Tensor
    base_index: torch.Tensor
    transform_index: torch.Tensor


class OrbitBufferDataset(Dataset):
    def __init__(self, records: Dict[str, torch.Tensor]) -> None:
        self.records = records
        self.length = int(records["label"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.records.items()}


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def _build_optimizer_and_scheduler(model: diffusion.Diffusion, config) -> Tuple[torch.optim.Optimizer, Any]:
    optimizer = torch.optim.AdamW(
        itertools.chain(model.backbone.parameters(), model.noise.parameters()),
        lr=config.optim.lr,
        betas=(config.optim.beta1, config.optim.beta2),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay,
    )
    scheduler = hydra.utils.instantiate(config.lr_scheduler, optimizer=optimizer)
    return optimizer, scheduler


@torch.no_grad()
def conditioned_sample(
    model: diffusion.Diffusion,
    start_x: torch.Tensor,
    num_steps: Optional[int] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Conditioned infill sampler for Sudoku anchors.

    Uses the repository's existing reverse updates and simply starts from
    a partially observed board instead of an all-mask prior.
    Because `_ddpm_update` and `_ddpm_caching_update` already preserve
    non-mask tokens via `copy_flag = (x != self.mask_index)`, anchors remain
    fixed through the trajectory. We re-clamp once more after noise removal.
    """
    was_training = model.training
    model.backbone.eval()
    model.noise.eval()

    if model.parameterization == "ar":
        raise NotImplementedError("Orbit-KTO is designed for masked diffusion, not AR mode.")

    if num_steps is None:
        num_steps = model.config.sampling.steps

    x = start_x.to(model.device).clone()
    fixed_mask = (x != model.mask_index)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = (1.0 - eps) / float(num_steps)
    p_x0_cache = None

    for i in range(num_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=model.device)
        if model.sampler == "ddpm":
            x = model._ddpm_update(x, t, dt)
        elif model.sampler == "ddpm_cache":
            p_x0_cache, x_next = model._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)
            if (not torch.allclose(x_next, x)) or model.time_conditioning:
                p_x0_cache = None
            x = x_next
        else:
            x = model._analytic_update(x, t, dt)

        x = torch.where(fixed_mask, start_x.to(model.device), x)

    if model.config.sampling.noise_removal:
        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=model.device)
        if model.sampler == "analytic":
            x_next = model._denoiser_update(x, t)
        else:
            unet_conditioning = model.noise(t)[0]
            x_next = model.forward(x, unet_conditioning).argmax(dim=-1)
        x = torch.where(fixed_mask, start_x.to(model.device), x_next)

    if was_training:
        model.backbone.train()
        model.noise.train()
    return x


def _fixed_t_token_loss(
    model: diffusion.Diffusion,
    x0: torch.Tensor,
    t: torch.Tensor,
    anchor_mask: Optional[torch.Tensor] = None,
    xt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Replica of `_forward_pass_diffusion` with user-specified t.

    Returns per-token diffusion loss with shape [B, L].
    """
    if model.T > 0:
        t = (t * model.T).to(torch.int)
        t = t / model.T
        t = t + (1.0 / model.T)

    if model.change_of_variables:
        unet_conditioning = t[:, None]
        f_T = torch.log1p(-torch.exp(-model.noise.sigma_max))
        f_0 = torch.log1p(-torch.exp(-model.noise.sigma_min))
        move_chance = torch.exp(f_0 + t * (f_T - f_0))
        move_chance = move_chance[:, None]
        sigma = None
        dsigma = None
    else:
        sigma, dsigma = model.noise(t)
        unet_conditioning = sigma[:, None]
        move_chance = 1 - torch.exp(-sigma[:, None])

    maskable_mask = None
    if anchor_mask is not None:
        maskable_mask = ~anchor_mask.bool()

    if xt is None:
        xt = model.q_xt(x0, move_chance, maskable_mask=maskable_mask)
    model_output = model.forward(xt, unet_conditioning)

    if model.parameterization == "sedd":
        if sigma is None:
            raise RuntimeError("SEDD path requires sigma")
        return dsigma[:, None] * model._score_entropy(model_output, sigma[:, None], xt, x0)

    if model.T > 0:
        diffusion_loss = model._d3pm_loss(model_output=model_output, xt=xt, x0=x0, t=t)
        if model.parameterization == "d3pm":
            reconstruction_loss = model._reconstruction_loss(x0)
        elif model.parameterization == "subs":
            reconstruction_loss = 0.0
        else:
            reconstruction_loss = 0.0
        return reconstruction_loss + diffusion_loss

    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1)
    if model.change_of_variables or model.importance_sampling:
        return -log_p_theta * torch.log1p(-torch.exp(-model.noise.sigma_min))
    assert sigma is not None and dsigma is not None
    return -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]


def _sample_xt(
    model: diffusion.Diffusion,
    x0: torch.Tensor,
    t: torch.Tensor,
    anchor_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample a single noised state x_t for shared policy/ref scoring."""
    t_local = t
    if model.T > 0:
        t_local = (t_local * model.T).to(torch.int)
        t_local = t_local / model.T
        t_local = t_local + (1.0 / model.T)

    if model.change_of_variables:
        f_T = torch.log1p(-torch.exp(-model.noise.sigma_max))
        f_0 = torch.log1p(-torch.exp(-model.noise.sigma_min))
        move_chance = torch.exp(f_0 + t_local * (f_T - f_0))
        move_chance = move_chance[:, None]
    else:
        sigma, _ = model.noise(t_local)
        move_chance = 1 - torch.exp(-sigma[:, None])

    maskable_mask = None
    if anchor_mask is not None:
        maskable_mask = ~anchor_mask.bool()
    return model.q_xt(x0, move_chance, maskable_mask=maskable_mask)


def completion_score(
    model: diffusion.Diffusion,
    x0: torch.Tensor,
    attention_mask: torch.Tensor,
    anchor_mask: Optional[torch.Tensor] = None,
    t_grid: Sequence[float] = (0.15, 0.35, 0.6, 0.9),
    mc_samples_per_t: int = 1,
) -> torch.Tensor:
    """Diffusion-native scalar score used by Orbit-KTO.

    Larger is better. We negate the average target-position diffusion loss
    over a fixed noising grid so that higher scores mean greater model
    compatibility with the candidate completion.
    """
    B = x0.shape[0]
    device = x0.device
    attn = attention_mask.float()
    denom = attn.sum(dim=-1).clamp_min(1.0)
    total = torch.zeros(B, device=device, dtype=torch.float32)

    for t_scalar in t_grid:
        t = torch.full((B,), float(t_scalar), device=device, dtype=torch.float32)
        accum = 0.0
        for _ in range(max(1, int(mc_samples_per_t))):
            tok_loss = _fixed_t_token_loss(model, x0, t, anchor_mask=anchor_mask).float()
            accum = accum + tok_loss
        tok_loss = accum / float(max(1, int(mc_samples_per_t)))
        total = total - (tok_loss * attn).sum(dim=-1) / denom

    return total / float(len(tuple(t_grid)))


def completion_score_pair_shared_xt(
    policy: diffusion.Diffusion,
    ref: diffusion.Diffusion,
    x0: torch.Tensor,
    attention_mask: torch.Tensor,
    anchor_mask: Optional[torch.Tensor] = None,
    t_grid: Sequence[float] = (0.15, 0.35, 0.6, 0.9),
    mc_samples_per_t: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute policy/ref scores against identical sampled x_t states."""
    B = x0.shape[0]
    device = x0.device
    attn = attention_mask.float()
    denom = attn.sum(dim=-1).clamp_min(1.0)
    total_pi = torch.zeros(B, device=device, dtype=torch.float32)
    total_ref = torch.zeros(B, device=device, dtype=torch.float32)

    for t_scalar in t_grid:
        t = torch.full((B,), float(t_scalar), device=device, dtype=torch.float32)
        accum_pi = 0.0
        accum_ref = 0.0
        for _ in range(max(1, int(mc_samples_per_t))):
            # Critical for KTO: compare both models on the exact same noisy state.
            xt = _sample_xt(policy, x0, t, anchor_mask=anchor_mask)
            tok_loss_pi = _fixed_t_token_loss(
                policy, x0, t, anchor_mask=anchor_mask, xt=xt
            ).float()
            with torch.no_grad():
                tok_loss_ref = _fixed_t_token_loss(
                    ref, x0, t, anchor_mask=anchor_mask, xt=xt
                ).float()
            accum_pi = accum_pi + tok_loss_pi
            accum_ref = accum_ref + tok_loss_ref

        tok_loss_pi = accum_pi / float(max(1, int(mc_samples_per_t)))
        tok_loss_ref = accum_ref / float(max(1, int(mc_samples_per_t)))
        total_pi = total_pi - (tok_loss_pi * attn).sum(dim=-1) / denom
        total_ref = total_ref - (tok_loss_ref * attn).sum(dim=-1) / denom

    return (
        total_pi / float(len(tuple(t_grid))),
        total_ref / float(len(tuple(t_grid))),
    )


def _load_policy(config, tokenizer, logger) -> diffusion.Diffusion:
    ckpt = getattr(config.orbit_kto, "init_checkpoint", None)
    if ckpt in (None, "", "null"):
        ckpt = config.eval.checkpoint_path if hasattr(config, "eval") else None

    if ckpt not in (None, "", "null"):
        logger.info(f"Loading Orbit-KTO policy init from checkpoint: {ckpt}")
        model = diffusion.Diffusion.load_from_checkpoint(
            ckpt,
            tokenizer=tokenizer,
            config=config,
            map_location="cpu",
        )
    else:
        logger.info("No init checkpoint provided; initializing policy from config.")
        model = diffusion.Diffusion(config, tokenizer=tokenizer)
    return model


@torch.no_grad()
def _build_orbit_buffer(
    policy: diffusion.Diffusion,
    base_loader: DataLoader,
    config,
    logger,
    max_base_examples: int,
) -> Dict[str, torch.Tensor]:
    device = policy.device
    transform_names = choose_transforms(
        int(config.orbit_kto.num_transforms),
        shuffle=bool(getattr(config.orbit_kto, "shuffle_transforms_each_refresh", False)),
    )
    M = len(transform_names)

    all_candidate = []
    all_solution = []
    all_attention_mask = []
    all_anchor_mask = []
    all_label = []
    all_w_pos = []
    all_w_neg = []
    all_base_index = []
    all_transform_index = []

    base_seen = 0
    global_base_offset = 0

    logger.info(
        "Building Orbit-KTO buffer with %d transforms per base and %d sampling steps.",
        M,
        int(config.orbit_kto.sample_steps),
    )

    policy.backbone.eval()
    policy.noise.eval()

    for batch in base_loader:
        batch = _move_batch(batch, device)
        solution = batch["input_ids"]
        anchor_mask = batch["anchor_mask"].bool()
        B = solution.shape[0]

        if base_seen >= max_base_examples:
            break
        if base_seen + B > max_base_examples:
            keep = max_base_examples - base_seen
            solution = solution[:keep]
            anchor_mask = anchor_mask[:keep]
            B = keep

        orbit = make_orbit_tensors(
            solution=solution,
            anchor_mask=anchor_mask,
            mask_index=policy.mask_index,
            transform_names=transform_names,
        )

        cand = conditioned_sample(
            policy,
            orbit["start_x"],
            num_steps=int(config.orbit_kto.sample_steps),
            eps=float(getattr(config.orbit_kto, "sampling_eps", 1e-5)),
        )
        label = exact_match(cand, orbit["solution"])
        local_base_index = orbit["base_index"]
        w_pos, w_neg = orbit_counterexample_weights(label, local_base_index, M)

        all_candidate.append(cand.cpu())
        all_solution.append(orbit["solution"].cpu())
        all_attention_mask.append(orbit["attention_mask"].cpu())
        all_anchor_mask.append(orbit["anchor_mask"].cpu())
        all_label.append(label.cpu())
        all_w_pos.append(w_pos.cpu())
        all_w_neg.append(w_neg.cpu())
        all_base_index.append((local_base_index + global_base_offset).cpu())
        all_transform_index.append(orbit["transform_index"].cpu())

        stats = orbit_statistics(label, local_base_index, M)
        logger.info(
            "refresh-buffer partial: bases=%d any=%.4f all=%.4f lsr=%.4f",
            B,
            float(stats["any"].cpu()),
            float(stats["all"].cpu()),
            float(stats["lsr"].cpu()),
        )

        global_base_offset += B
        base_seen += B

    if base_seen == 0:
        raise RuntimeError("Orbit-KTO buffer build produced zero base examples.")

    return {
        "candidate": torch.cat(all_candidate, dim=0),
        "solution": torch.cat(all_solution, dim=0),
        "attention_mask": torch.cat(all_attention_mask, dim=0),
        "anchor_mask": torch.cat(all_anchor_mask, dim=0),
        "label": torch.cat(all_label, dim=0).bool(),
        "w_pos": torch.cat(all_w_pos, dim=0).float(),
        "w_neg": torch.cat(all_w_neg, dim=0).float(),
        "base_index": torch.cat(all_base_index, dim=0).long(),
        "transform_index": torch.cat(all_transform_index, dim=0).long(),
    }


def _orbit_kto_loss(
    policy: diffusion.Diffusion,
    ref: diffusion.Diffusion,
    batch: Dict[str, torch.Tensor],
    config,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    label = batch["label"].float()
    w_pos = batch["w_pos"].float()
    w_neg = batch["w_neg"].float()

    score_pi, score_ref = completion_score_pair_shared_xt(
        policy=policy,
        ref=ref,
        x0=batch["candidate"],
        attention_mask=batch["attention_mask"],
        anchor_mask=batch["anchor_mask"],
        t_grid=tuple(config.orbit_kto.score_t_grid),
        mc_samples_per_t=int(config.orbit_kto.mc_samples_per_t),
    )

    delta = score_pi - score_ref

    alpha_pos = float(config.orbit_kto.alpha_pos)
    alpha_neg = float(config.orbit_kto.alpha_neg)
    pos_mass = alpha_pos * w_pos * label
    neg_mass = alpha_neg * w_neg * (1.0 - label)
    active = pos_mass + neg_mass
    # Optionally normalize delta to reduce diffusion-score scale variance.
    if bool(getattr(config.orbit_kto, "normalize_delta", False)):
        w_sum = active.sum() + 1e-8
        d_mean = (active * delta.detach()).sum() / w_sum
        d_var = (active * (delta.detach() - d_mean).pow(2)).sum() / w_sum
        d_std = torch.sqrt(d_var + 1e-8)
        delta = (delta - d_mean) / d_std
    else:
        d_std = delta.detach().std(unbiased=False)

    kappa = (active * delta.detach()).sum() / (active.sum() + 1e-8)
    # Keep old behavior when kappa_min=0.0; allow disabling bias with null.
    kappa_min = getattr(config.orbit_kto, "kappa_min", 0.0)
    if kappa_min is not None:
        kappa = torch.clamp(kappa, min=float(kappa_min))

    tau = float(config.orbit_kto.tau)
    pos_loss = F.softplus(-tau * (delta - kappa))
    neg_loss = F.softplus(tau * (delta - kappa))

    kto_loss = (pos_mass * pos_loss + neg_mass * neg_loss).sum() / (active.sum() + 1e-8)

    stats = {
        "score_pi": score_pi.detach().mean(),
        "score_ref": score_ref.detach().mean(),
        "delta": delta.detach().mean(),
        "delta_std": d_std.detach(),
        "kappa": kappa.detach(),
        "active_mass": active.detach().sum(),
        "pos_mass": pos_mass.detach().sum(),
        "neg_mass": neg_mass.detach().sum(),
    }
    return kto_loss, stats


def _ac_aux_loss(
    policy: diffusion.Diffusion,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    losses = policy._loss(
        batch["solution"],
        batch["attention_mask"],
        anchor_mask=batch["anchor_mask"],
    )
    return losses.loss


def _save_checkpoint(policy: diffusion.Diffusion, optimizer, scheduler, step: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"orbit_kto_step{step}.pt"
    payload = {
        "step": step,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "ema_state_dict": policy.ema.state_dict() if getattr(policy, "ema", None) is not None else None,
    }
    torch.save(payload, path)
    return path


def _prune_old_checkpoints(output_dir: Path, max_to_keep: Optional[int], logger) -> None:
    """Keep only the newest Orbit-KTO checkpoints to bound disk usage."""
    if max_to_keep is None:
        return
    max_to_keep = int(max_to_keep)
    if max_to_keep <= 0:
        return

    ckpts = sorted(output_dir.glob("orbit_kto_step*.pt"), key=lambda p: p.stat().st_mtime)
    if len(ckpts) <= max_to_keep:
        return

    to_delete = ckpts[: len(ckpts) - max_to_keep]
    for p in to_delete:
        try:
            p.unlink()
        except OSError as exc:
            logger.warning("Failed to delete old checkpoint %s: %s", p, exc)


def train(config, logger, tokenizer):
    """Entry point called from main.py with `mode=orbit_kto`."""
    if config.data.train != "sudoku9-anchors" or config.data.valid != "sudoku9-anchors":
        logger.warning(
            "Orbit-KTO works best with data=sudoku9-anchors. "
            "Current config uses train=%s valid=%s",
            config.data.train,
            config.data.valid,
        )

    train_loader, valid_loader = dataloader.get_dataloaders(config, tokenizer)
    policy = _load_policy(config, tokenizer, logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    if getattr(policy, "ema", None) is not None:
        policy.ema.move_shadow_params_to_device(device)
    policy.train()

    ref = copy.deepcopy(policy).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    optimizer, scheduler = _build_optimizer_and_scheduler(policy, config)

    examples_per_refresh = int(config.orbit_kto.examples_per_refresh)
    updates_per_refresh = int(config.orbit_kto.updates_per_refresh)
    orbit_batch_size = int(config.orbit_kto.batch_size)
    max_steps = int(config.orbit_kto.max_steps)
    log_every = int(getattr(config.orbit_kto, "log_every", 20))
    save_every = int(getattr(config.orbit_kto, "save_every", 500))
    max_to_keep = getattr(config.orbit_kto, "max_checkpoints_to_keep", 3)
    output_dir = Path(getattr(config.orbit_kto, "output_dir", "outputs/orbit_kto"))
    lambda_ac = float(getattr(config.orbit_kto, "lambda_ac", 0.0))

    global_step = 0
    refresh_round = 0

    while global_step < max_steps:
        refresh_round += 1
        logger.info("=== Orbit-KTO refresh round %d ===", refresh_round)
        buffer_dict = _build_orbit_buffer(
            policy=policy,
            base_loader=train_loader,
            config=config,
            logger=logger,
            max_base_examples=examples_per_refresh,
        )
        dataset = OrbitBufferDataset(buffer_dict)
        buffer_loader = DataLoader(
            dataset,
            batch_size=orbit_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        for local_u, batch in enumerate(buffer_loader):
            if global_step >= max_steps or local_u >= updates_per_refresh:
                break

            batch = _move_batch(batch, device)
            policy.train()
            optimizer.zero_grad(set_to_none=True)

            kto_loss, stats = _orbit_kto_loss(policy, ref, batch, config)
            if lambda_ac > 0:
                ac_loss = _ac_aux_loss(policy, batch)
            else:
                ac_loss = torch.zeros((), device=device)

            loss = kto_loss + lambda_ac * ac_loss
            loss.backward()

            grad_clip = getattr(config.optim, "grad_clip", None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(policy.backbone.parameters(), policy.noise.parameters()),
                    max_norm=float(grad_clip),
                )

            optimizer.step()
            if getattr(policy, "ema", None) is not None:
                policy.ema.update(itertools.chain(
                    policy.backbone.parameters(),
                    policy.noise.parameters(),
                ))
            if scheduler is not None:
                scheduler.step()

            global_step += 1

            if global_step % log_every == 0 or global_step == 1:
                logger.info(
                    "step=%d refresh=%d kto_loss=%.6f ac_loss=%.6f total=%.6f "
                    "delta=%.4f delta_std=%.4f kappa=%.4f active=%.1f pos=%.1f neg=%.1f",
                    global_step,
                    refresh_round,
                    float(kto_loss.detach().cpu()),
                    float(ac_loss.detach().cpu()),
                    float(loss.detach().cpu()),
                    float(stats["delta"].detach().cpu()),
                    float(stats["delta_std"].detach().cpu()),
                    float(stats["kappa"].detach().cpu()),
                    float(stats["active_mass"].detach().cpu()),
                    float(stats["pos_mass"].detach().cpu()),
                    float(stats["neg_mass"].detach().cpu()),
                )

            if global_step % save_every == 0:
                ckpt = _save_checkpoint(policy, optimizer, scheduler, global_step, output_dir)
                _prune_old_checkpoints(output_dir, max_to_keep=max_to_keep, logger=logger)
                logger.info("Saved Orbit-KTO checkpoint to %s", ckpt)

    final_ckpt = _save_checkpoint(policy, optimizer, scheduler, global_step, output_dir)
    _prune_old_checkpoints(output_dir, max_to_keep=max_to_keep, logger=logger)
    logger.info("Orbit-KTO finished at step %d. Final checkpoint: %s", global_step, final_ckpt)
