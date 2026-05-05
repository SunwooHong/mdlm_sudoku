import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist
import os
import sys
import math
from collections import defaultdict

# Ensure KLASS/src is importable when running from d1/SFT.
_THIS_DIR = os.path.dirname(__file__)

# Make `from utils import ...` work (KLASS/src).
_SEOHYUN_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_KLASS_SRC = os.path.join(_SEOHYUN_ROOT, "KLASS", "src")
if _KLASS_SRC not in sys.path:
    sys.path.insert(0, _KLASS_SRC)

# Make `from model.llada_klass import ...` work.
# Insert *after* KLASS so this repo's `model/` wins over `KLASS/src/model/`.
_LLaDA_PARENT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _LLaDA_PARENT not in sys.path:
    sys.path.insert(0, _LLaDA_PARENT)

from model.llada_klass import stable_confident_decode
from utils import extract_math_answer, compare_answers

from sft_trainer_d4_grouped import (
    ORBIT_TRANSFORMS,
    _ORBIT_INV_NP,
    _ORBIT_PERM_NP,
    _orbit_canonical_solution_token_ids,
    _transform_flat_sudoku,
    _build_assistant_digit_positions,
    _digit_vocab_ids,
)

_PAIR_GROUP = 2  # two D4 views per canonical puzzle
# Mix Trainer epoch into collate-time RNG without colliding across puzzles.
_PAIR_ORBIT_EPOCH_PRIME = 1_009_963_391


def _pair_orbit_effective_gamma(
    *,
    gamma: float,
    global_step: int,
    max_steps: int,
    warmup_frac: float,
    ramp_frac: float,
) -> float:
    """Scale ``γ`` for JS consistency: CE-only early, then linear ramp to ``gamma``.

    Uses ``TrainingArguments.max_steps`` as the horizon (same unit as ``global_step``).
    """
    gamma = float(gamma)
    if gamma <= 0.0:
        return 0.0
    max_steps = int(max_steps)
    if max_steps <= 0:
        return gamma
    wu = max(0.0, min(1.0, float(warmup_frac)))
    ramp = max(0.0, min(1.0, float(ramp_frac)))
    wu_end = int(round(wu * max_steps))
    ramp_end = int(round((wu + ramp) * max_steps))
    gs = int(global_step)
    if gs < wu_end:
        return 0.0
    if ramp <= 0.0 or ramp_end <= wu_end or gs >= ramp_end:
        return gamma
    t = float(gs - wu_end) / float(max(ramp_end - wu_end, 1))
    return gamma * max(0.0, min(1.0, t))


def _pair_orbit_resolve_training_steps(trainer) -> int:
    """Total update horizon for γ warmup (``max_steps`` or HF-computed ``state.max_steps``)."""
    args = trainer.args
    ms = int(getattr(args, "max_steps", -1) or -1)
    if ms > 0:
        return ms
    sms = getattr(trainer.state, "max_steps", None)
    if sms is not None and int(sms) > 0:
        return int(sms)
    try:
        dl = trainer.get_train_dataloader()
        steps_per_epoch = len(dl)
    except Exception:
        steps_per_epoch = 0
    ne = float(getattr(args, "num_train_epochs", 1) or 1)
    return max(1, int(math.ceil(steps_per_epoch * ne)))


def _pair_orbit_tokenize_one_arm(
    puzzle_canonical_81: str,
    solution_canonical_81: str,
    tf_idx: int,
    tokenizer,
    max_length: int,
    prompt_style: str,
    few_shot: bool,
    orig_idx: int,
    arm: int,
    pair_mask_seed: int,
) -> dict | None:
    """Tokenize one D4 view for pair-orbit training; returns None on skip (same rules as preprocess)."""
    puzzle = str(puzzle_canonical_81).strip()
    solution = str(solution_canonical_81).strip()
    if not puzzle or not solution:
        return None
    try:
        n = int(math.isqrt(len(puzzle)))
        if n * n != len(puzzle):
            return None
        blank_chars = {".", "0", "_"}
        canon_blank = torch.tensor([puzzle[j] in blank_chars for j in range(n * n)], dtype=torch.bool)
        if int(canon_blank.sum().item()) == 0:
            return None

        canon_solution_fmt = format_sudoku_grid(solution)
        canon_tgt_ids = _orbit_canonical_solution_token_ids(tokenizer, canon_solution_fmt, n)
        tf_name = ORBIT_TRANSFORMS[tf_idx]
        p_tf = _transform_flat_sudoku(puzzle, tf_name)
        s_tf = _transform_flat_sudoku(solution, tf_name)
        sub = {"puzzle": p_tf, "solution": s_tf}

        puzzle_text_fmt = format_sudoku_grid(p_tf)
        solution_text_fmt = format_sudoku_grid(s_tf)

        prompt = [{"role": "user", "content": build_sudoku_prompt(sub, prompt_style=prompt_style, few_shot=few_shot)}]
        response = [{"role": "assistant", "content": solution_text_fmt}]
        inputs, prompt_text = build_chat_example(tokenizer, prompt, response)

        tokenized_input = tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ).input_ids.squeeze(0)
        tokenized_prompt = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            non_pad_len = int((tokenized_input != pad_id).sum().item())
        else:
            non_pad_len = int(tokenized_input.shape[0])
        start_search = int(tokenized_prompt.attention_mask.sum(-1).item())
        if start_search >= non_pad_len:
            return None

        blank_token_mask = torch.zeros((max_length,), dtype=torch.bool)
        _, digit_id_set = _digit_vocab_ids(tokenizer, n)
        digit_positions_in_assistant = [
            pos for pos in range(start_search, max_length) if int(tokenized_input[pos].item()) in digit_id_set
        ]
        expected_digits = n * n
        if len(digit_positions_in_assistant) < expected_digits:
            return None
        digit_positions_in_assistant = digit_positions_in_assistant[:expected_digits]

        blank_cells_tf = [idx for idx, ch in enumerate(p_tf) if ch in blank_chars]
        for cell_idx in blank_cells_tf:
            blank_token_mask[int(digit_positions_in_assistant[cell_idx])] = True

        if blank_token_mask.sum().item() == 0:
            return None

        assist_pos, _ = _build_assistant_digit_positions(
            tokenized_input,
            tokenized_prompt.attention_mask.sum(-1),
            tokenizer,
            puzzle_81=p_tf,
            max_length=max_length,
        )

        attn_mask = torch.ones_like(tokenized_input)
        if pad_id is not None:
            attn_mask = (tokenized_input != pad_id).long()

        return {
            "input_ids": tokenized_input,
            "attention_mask": attn_mask,
            "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
            "orig_idx": orig_idx,
            "prompt_text": prompt_text,
            "solution_text": solution_text_fmt,
            "pair_canon_target_ids": canon_tgt_ids.clone(),
            "pair_assist_digit_pos": assist_pos.clone(),
            "pair_group_idx": orig_idx,
            "pair_arm": arm,
            "pair_view_idx": tf_idx,
            "pair_mask_seed": pair_mask_seed,
            "pair_canon_blank_cell": canon_blank.clone(),
            "pair_canon_puzzle_flat": puzzle,
            "pair_canon_solution_flat": solution,
        }
    except Exception:
        return None


class PairOrbitEpochCallback(TrainerCallback):
    """Sets collator epoch so masking (and optional D4 pair) can change each HF epoch."""

    def __init__(self, collator):
        self.collator = collator

    def on_train_begin(self, args, state, control, **kwargs):
        if hasattr(self.collator, "pair_orbit_train_epoch"):
            self.collator.pair_orbit_train_epoch = 0
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        if hasattr(self.collator, "pair_orbit_train_epoch"):
            self.collator.pair_orbit_train_epoch = int(state.epoch)
        return control


def _sudoku_block_padded_gen_length(suffix_len: int, block_length: int, max_gen_length: int) -> int:
    """Pad assistant suffix length up to a multiple of ``block_length`` (must fit in ``max_gen_length``)."""
    gen_len = ((suffix_len + block_length - 1) // block_length) * block_length
    max_aligned = (max_gen_length // block_length) * block_length
    if gen_len > max_aligned:
        raise ValueError(
            f"Padded Sudoku assistant length {gen_len} exceeds max_gen_length-aligned cap {max_aligned} "
            f"(suffix_len={suffix_len}, block_length={block_length}, max_gen_length={max_gen_length})."
        )
    return int(gen_len)


def _sudoku_steps_divisible_by_blocks(requested_steps: int, num_blocks: int) -> int:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    steps = (max(requested_steps, num_blocks) // num_blocks) * num_blocks
    if steps < num_blocks:
        steps = num_blocks
    return int(steps)


class dLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        """
        labels, t, num_prompt_tokens = inputs.pop("labels"), inputs.pop("t"), inputs.pop("num_prompt_tokens")
        outputs = model(**inputs)
        logits = outputs.logits
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({"unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item()})
        # Only masked positions have labels != -100; keep loss normalized over them.
        non_ignored = (labels != -100).sum().clamp_min(1)
        loss = (unscaled_loss / t).sum() / non_ignored
        return loss if not return_outputs else (loss, outputs)

    @torch.no_grad()
    def _compute_validation_accuracy(self, eval_dataset, max_samples: int = 50):
        """
        Validation accuracy computed by:
        prompt/question -> stable_confident_decode -> extract answer -> compare_answers
        """
        if eval_dataset is None:
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        model = self.model
        model.eval()

        tokenizer = getattr(self.data_collator, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Need data_collator.tokenizer for validation decoding.")

        correct = 0
        total = 0
        seen_orig = set()

        device = next(model.parameters()).device
        decode_gen_length = getattr(self.args, "eval_decode_gen_length", 512)
        decode_steps = getattr(self.args, "eval_decode_steps", 512)
        decode_block_length = getattr(self.args, "eval_decode_block_length", 512)
        decode_temperature = getattr(self.args, "eval_decode_temperature", 0.0)
        decode_conf_threshold = getattr(self.args, "eval_decode_conf_threshold", 0.9)
        decode_kl_threshold = getattr(self.args, "eval_decode_kl_threshold", 0.01)
        decode_kl_history_length = getattr(self.args, "eval_decode_kl_history_length", 2)
        decode_alg = getattr(self.args, "eval_decode_alg", "default")
        decode_unmask_strategy = getattr(self.args, "eval_decode_unmask_strategy", "all")
        mask_id = getattr(self.args, "eval_decode_mask_id", 126336)

        for idx in range(min(len(eval_dataset), max_samples)):
            item = eval_dataset[idx]
            orig_idx = item.get("orig_idx", idx)
            if orig_idx in seen_orig:
                continue
            seen_orig.add(orig_idx)

            prompt_text = item.get("prompt_text")
            if prompt_text is None:
                question = item["question"]
                full_trajectory = item["full_trajectory"]
                prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]
                prompt_text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
                target_text = full_trajectory
                use_grid_match = False
            else:
                target_text = item["solution_text"]
                use_grid_match = True

            if use_grid_match and "input_ids" in item and "blank_token_mask" in item and "prompt_lengths" in item:
                # Full partially-filled sequence (prompt + anchored assistant + [MASK] blanks): decode in-place.
                pad_id = tokenizer.pad_token_id
                full_ids = item["input_ids"].to(device).long().clone()
                bm = item["blank_token_mask"].to(device).bool()
                if bm.shape != full_ids.shape:
                    raise ValueError("blank_token_mask must align with input_ids for Sudoku eval.")
                full_ids[bm] = mask_id
                if pad_id is not None:
                    eff = int((full_ids != pad_id).sum().item())
                else:
                    eff = int(full_ids.numel())
                ids_e = full_ids[:eff]
                pl = int(item["prompt_lengths"].item()) if torch.is_tensor(item["prompt_lengths"]) else int(item["prompt_lengths"])
                suffix_len = int(ids_e.numel() - pl)

                ids_row = ids_e.unsqueeze(0)
                L = int(ids_row.shape[1])
                pad_len = (-L) % decode_block_length
                if pad_len:
                    block_pad_id = tokenizer.eos_token_id
                    if block_pad_id is None:
                        block_pad_id = mask_id
                    pad = torch.full((1, pad_len), block_pad_id, device=device, dtype=ids_row.dtype)
                    ids_padded = torch.cat([ids_row, pad], dim=1)
                else:
                    ids_padded = ids_row

                # Block padding can extend past training `max_length`; only hard-limit by tokenizer context.
                padded_len = int(ids_padded.shape[1])
                tok_max = getattr(tokenizer, "model_max_length", None)
                if tok_max is not None and int(tok_max) < 10**9 and padded_len > int(tok_max):
                    raise ValueError(
                        f"Padded Sudoku eval sequence length {padded_len} exceeds tokenizer.model_max_length={int(tok_max)}."
                    )

                num_blocks = ids_padded.shape[1] // decode_block_length
                decode_steps_use = _sudoku_steps_divisible_by_blocks(decode_steps, num_blocks)
                dummy_prefix = torch.empty((1, 0), dtype=torch.long, device=device)

                x_output, _used_steps = stable_confident_decode(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids_original=dummy_prefix,
                    gen_length=0,
                    steps=decode_steps_use,
                    block_length=decode_block_length,
                    temperature=decode_temperature,
                    mask_id=mask_id,
                    conf_threshold=decode_conf_threshold,
                    kl_threshold=decode_kl_threshold,
                    kl_history_length=decode_kl_history_length,
                    alg=decode_alg,
                    unmask_strategy=decode_unmask_strategy,
                    partial_sequence=ids_padded,
                )
                gen_slice = x_output[:, pl : pl + suffix_len]
                generated_text = tokenizer.batch_decode(gen_slice, skip_special_tokens=True)[0]
            else:
                input_ids_original = torch.tensor(tokenizer(prompt_text)["input_ids"], device=device).unsqueeze(0)

                x_output, _used_steps = stable_confident_decode(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids_original=input_ids_original,
                    gen_length=decode_gen_length,
                    steps=decode_steps,
                    block_length=decode_block_length,
                    temperature=decode_temperature,
                    mask_id=mask_id,
                    conf_threshold=decode_conf_threshold,
                    kl_threshold=decode_kl_threshold,
                    kl_history_length=decode_kl_history_length,
                    alg=decode_alg,
                    unmask_strategy=decode_unmask_strategy,
                )

                generated_text = tokenizer.batch_decode(
                    x_output[:, input_ids_original.shape[1] :], skip_special_tokens=True
                )[0]

            if use_grid_match:
                is_correct = canonical_sudoku_digits(generated_text) == canonical_sudoku_digits(target_text)
            else:
                ground_truth_answer = extract_math_answer(prompt_text, target_text)
                generated_answer = extract_math_answer(prompt_text, generated_text)
                is_correct = compare_answers(prompt_text, ground_truth_answer, generated_answer)
            if is_correct:
                correct += 1
            total += 1

            # import pdb; pdb.set_trace() 
        return {"accuracy": (correct / total if total else 0.0), "correct": correct, "total": total}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        print(f"[dLLMTrainer.evaluate] custom evaluate() running from {__file__}")
        metrics = super().evaluate(
            eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        is_zero = not hasattr(self, "is_world_process_zero") or self.is_world_process_zero()
        if is_zero:
            val_acc = self._compute_validation_accuracy(eval_dataset or self.eval_dataset)
            acc_percent = float(round(val_acc["accuracy"] * 100, 2))
        else:
            acc_percent = 0.0

        if dist.is_available() and dist.is_initialized():
            # NCCL (GPU DDP) cannot broadcast CPU tensors.
            bcast_dev = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            t = torch.tensor([acc_percent], dtype=torch.float32, device=bcast_dev)
            dist.broadcast(t, src=0)
            acc_percent = float(t.item())

        metrics[f"{metric_key_prefix}_accuracy"] = acc_percent
        if is_zero:
            self.log({f"{metric_key_prefix}_accuracy": metrics[f"{metric_key_prefix}_accuracy"]})
        return metrics


def _js_divergence_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """JS(p,q) per row; p,q are probabilities [N, K]."""
    m = 0.5 * (p + q)
    lp = p.clamp(min=eps).log()
    lq = q.clamp(min=eps).log()
    lm = m.clamp(min=eps).log()
    kl_p = (p * (lp - lm)).sum(dim=-1)
    kl_q = (q * (lq - lm)).sum(dim=-1)
    return 0.5 * (kl_p + kl_q)


class PairOrbitDistributedSampler(torch.utils.data.Sampler[int]):
    """DDP: shards by whole pair blocks (2 consecutive rows per puzzle)."""

    def __init__(self, total_rows: int, num_replicas: int, rank: int, *, seed: int = 0, drop_last: bool = False):
        if total_rows % _PAIR_GROUP != 0:
            raise ValueError(
                f"PairOrbitDistributedSampler: dataset length {total_rows} must be divisible by {_PAIR_GROUP}."
            )
        self.num_groups = total_rows // _PAIR_GROUP
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        if self.num_replicas <= 0 or self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"Invalid num_replicas={self.num_replicas} rank={self.rank}")
        if self.num_groups % self.num_replicas != 0:
            raise ValueError(
                f"Pair-orbit DDP: number of puzzle groups ({self.num_groups}) must be divisible by "
                f"world_size ({self.num_replicas})."
            )
        self._indices: list[int] = []
        for g in range(self.num_groups):
            if g % self.num_replicas != self.rank:
                continue
            base = g * _PAIR_GROUP
            self._indices.extend(range(base, base + _PAIR_GROUP))

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def set_epoch(self, epoch: int) -> None:
        _ = epoch


class dLLMPairOrbitTrainer(dLLMTrainer):
    """Pairwise Orbit Consistency SFT (memory-friendly LE-oriented post-train).

    Two D4 views ``g``, ``h`` of the same canonical state share the same canonical blank mask
    and the same diffusion timestep ``t`` (see ``PairOrbitSudokuCollator``). Logits are pulled
    back to canonical cell order before CE and JS.

    **Loss** (paper form; this trainer matches it up to diffusion scaling on CE)::

      L = (1/2)[CE(g) + CE(h)] + γ * (1/|S|) * Σ_{i∈S} JS(p_g(i), p_h(i))

    where ``S`` is the set of supervised blank cells in canonical coordinates, and
    ``p_g(i)``, ``p_h(i)`` are 9-way softmaxes over digit vocab at cell ``i``.
    JS is symmetric vs one-way KL.

    **Diffusion alignment:** CE terms are divided by ``t_b`` like ``dLLMTrainer``; the JS term
    is *not* divided by ``t`` (matches the usual "auxiliary consistency" pattern).

    **Hyperparameters (suggested):** set ``pair_orbit_gamma`` on ``TrainingArguments`` to ``0.01``
    (probe), ``0.05`` (main), or ``0.1`` only if stable (defaults to ``1.0`` if unset — override).
    **Warmup (recommended):** ``pair_orbit_consistency_warmup_frac=0.15`` and
    ``pair_orbit_consistency_ramp_frac=0.10`` (defaults ``0`` = no ramp, prior behavior).
    Consistency alone can agree on wrong digits — always keep the gold CE branch on.
    """

    def _get_train_sampler(self):
        if self.train_dataset is None or len(self.train_dataset) == 0:
            return super()._get_train_sampler()
        ex = self.train_dataset[0]
        if not isinstance(ex, dict) or ex.get("pair_arm") is None:
            return super()._get_train_sampler()
        from torch.utils.data import SequentialSampler

        ws, rank = 1, 0
        if dist.is_available() and dist.is_initialized():
            ws = int(dist.get_world_size())
            rank = int(dist.get_rank())
        else:
            _ws_env = os.environ.get("WORLD_SIZE", "").strip()
            if _ws_env.isdigit():
                ws = max(int(_ws_env), 1)
                rank = int(os.environ.get("RANK", "0"))
            else:
                ws = int(getattr(self.args, "world_size", 1) or 1)
                if ws > 1:
                    rank = int(os.environ.get("RANK", getattr(self.args, "local_rank", 0)))

        if ws <= 1:
            return SequentialSampler(self.train_dataset)

        return PairOrbitDistributedSampler(
            len(self.train_dataset),
            num_replicas=ws,
            rank=rank,
            seed=int(getattr(self.args, "seed", 0) or 0),
        )

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        ex0 = self.train_dataset[0] if len(self.train_dataset) else None
        if ex0 is None or not isinstance(ex0, dict) or ex0.get("pair_arm") is None:
            return super().get_train_dataloader()

        from functools import partial

        from torch.utils.data import DataLoader
        from transformers.trainer_utils import has_length

        try:
            from transformers.trainer_utils import seed_worker
        except ImportError:  # pragma: no cover
            from transformers.trainer_pt_utils import seed_worker

        if not has_length(self.train_dataset):
            return super().get_train_dataloader()

        dataset = self.train_dataset
        data_collator = self._get_collator_with_removed_columns(self.data_collator, description="Training")

        coll0 = self.data_collator
        num_workers = int(self.args.dataloader_num_workers)
        if (
            isinstance(coll0, PairOrbitSudokuCollator)
            and getattr(coll0, "pair_orbit_refresh_each_epoch", False)
            and num_workers > 0
        ):
            # Workers are forked with a stale collator; epoch updates would not reach them.
            num_workers = 0
        should_fork = torch.backends.mps.is_available() and num_workers > 1
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "multiprocessing_context": "fork" if should_fork else None,
            "sampler": self._get_train_sampler(),
            "drop_last": self.args.dataloader_drop_last,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            "worker_init_fn": partial(seed_worker, num_workers=num_workers, rank=self.args.process_index),
        }
        return DataLoader(dataset, **dataloader_params)

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        if "pair_B" not in inputs:
            return super().compute_loss(model, inputs, num_items_in_batch=num_items_in_batch, return_outputs=return_outputs)

        supervise_canon = inputs.pop("pair_supervise_canon")
        canon_target_flat = inputs.pop("pair_canon_target_ids")
        assist_pos_flat = inputs.pop("pair_assist_digit_pos")
        inv_cpu = inputs.pop("pair_inv_cell_perm")
        _pair_b = int(inputs.pop("pair_B").view(-1)[0].item())
        cluster_id = inputs.pop("pair_cluster_id").long()
        view_flat = inputs.pop("pair_view_idx_batch").long()
        inputs.pop("labels", None)
        t_flat = inputs.pop("t")
        inputs.pop("num_prompt_tokens", None)

        gamma = float(getattr(self.args, "pair_orbit_gamma", 1.0))
        wu_frac = float(getattr(self.args, "pair_orbit_consistency_warmup_frac", 0.0))
        ramp_frac = float(getattr(self.args, "pair_orbit_consistency_ramp_frac", 0.0))
        if getattr(self, "_pair_orbit_horizon_steps", None) is None:
            self._pair_orbit_horizon_steps = int(_pair_orbit_resolve_training_steps(self))
        max_steps = int(self._pair_orbit_horizon_steps)
        gamma_eff = _pair_orbit_effective_gamma(
            gamma=gamma,
            global_step=int(self.state.global_step),
            max_steps=max_steps,
            warmup_frac=wu_frac,
            ramp_frac=ramp_frac,
        )

        outputs = model(**inputs)
        logits = outputs.logits
        device = logits.device
        v_dim = int(logits.shape[-1])

        inv = inv_cpu.to(device=device, dtype=torch.long)
        supervise_canon = supervise_canon.to(device=device).float()
        canon_ref = canon_target_flat.to(device=device).long()
        assist_flat = assist_pos_flat.to(device=device).long()
        cluster_id = cluster_id.to(device=device)
        view_flat = view_flat.to(device=device)
        t_flat = t_flat.to(device=device).float()

        digit_ids = getattr(self, "_pair_orbit_digit_vocab", None)
        if digit_ids is None or digit_ids.device != device:
            ids9, _ = _digit_vocab_ids(self.tokenizer, 9)
            digit_ids = torch.tensor(ids9, device=device, dtype=torch.long)
            self._pair_orbit_digit_vocab = digit_ids

        n_rows = int(logits.shape[0])
        if int(cluster_id.shape[0]) != n_rows:
            raise ValueError("pair_cluster_id length must match logits batch dim.")

        c_count = int(_pair_b)
        if c_count != int(supervise_canon.shape[0]):
            raise ValueError("pair_B must match pair_supervise_canon leading dim.")

        loss_terms: list[torch.Tensor] = []

        for c in range(c_count):
            row_m = cluster_id == c
            ridx = torch.nonzero(row_m, as_tuple=False).squeeze(-1)
            if ridx.numel() != _PAIR_GROUP:
                raise ValueError(f"Pair cluster {c}: expected {_PAIR_GROUP} rows, got {ridx.numel()}.")

            ri0, ri1 = int(ridx[0].item()), int(ridx[1].item())
            v0, v1 = int(view_flat[ri0].item()), int(view_flat[ri1].item())

            def logits_canon_row(ri: int, vi: int) -> torch.Tensor:
                lg = torch.gather(
                    logits[ri : ri + 1],
                    1,
                    assist_flat[ri : ri + 1].unsqueeze(-1).expand(-1, 81, v_dim),
                )
                idx = inv[vi].view(1, 81, 1).expand(1, 81, v_dim)
                return torch.gather(lg, 1, idx).squeeze(0).contiguous().float()

            log_v0 = logits_canon_row(ri0, v0)
            log_v1 = logits_canon_row(ri1, v1)
            canon_t = canon_ref[ri0].contiguous()

            supervise = supervise_canon[c]
            denom = supervise.sum().clamp(min=1e-8)

            ce_a = (F.cross_entropy(log_v0, canon_t, reduction="none") * supervise).sum() / denom
            ce_b = (F.cross_entropy(log_v1, canon_t, reduction="none") * supervise).sum() / denom
            L_ce = 0.5 * (ce_a + ce_b)

            pg = F.softmax(torch.index_select(log_v0, 1, digit_ids), dim=-1)
            ph = F.softmax(torch.index_select(log_v1, 1, digit_ids), dim=-1)
            js_cell = _js_divergence_probs(pg, ph)
            L_js = (js_cell * supervise).sum() / denom

            t_min = float(getattr(self.args, "pair_orbit_t_min", 0.05))
            t_b = t_flat[ri0, 0].clamp(min=t_min)
            loss_terms.append((L_ce / t_b) + gamma_eff * L_js)

        loss_acc = torch.stack(loss_terms).mean()

        if self.args.logging_steps and (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(
                {
                    "pair_orbit_total": loss_acc.item(),
                    "pair_orbit_gamma_eff": float(gamma_eff),
                }
            )

        return loss_acc if not return_outputs else (loss_acc, outputs)


class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timesteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        # Standard behavior masks arbitrary token positions.
        # For Sudoku SFT we optionally restrict masking to blank cells only.
        rand = torch.rand((B, N), device=input_ids.device)
        mask_indices = rand < t
        if "blank_token_mask" in batch:
            blank_token_mask = batch["blank_token_mask"].to(device=input_ids.device, dtype=torch.bool)
            if blank_token_mask.shape != mask_indices.shape:
                raise ValueError(
                    f"blank_token_mask shape {blank_token_mask.shape} does not match input_ids shape {mask_indices.shape}"
                )
            mask_indices = mask_indices & blank_token_mask
            # Safety fallback: if a sample has blank positions but none were
            # selected by Bernoulli(t), force-mask one blank token so it
            # contributes a training signal.
            has_blank = blank_token_mask.any(dim=1)
            has_mask = mask_indices.any(dim=1)
            need_fallback = has_blank & (~has_mask)
            if need_fallback.any():
                for b in torch.nonzero(need_fallback, as_tuple=False).squeeze(-1).tolist():
                    blank_positions = torch.nonzero(blank_token_mask[b], as_tuple=False).squeeze(-1)
                    chosen_idx = blank_positions[torch.randint(0, blank_positions.numel(), (1,), device=input_ids.device)]
                    mask_indices[b, chosen_idx] = True
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        clean_batch = []
        for f in batch:
            clean_features = {}
            for k, v in f.items():
                if isinstance(v, torch.Tensor):
                    clean_features[k] = v
                elif isinstance(v, (int, float, bool)) or v is None:
                    if k not in {"orig_idx"}:
                        clean_features[k] = v
                else:
                    continue
            clean_batch.append(clean_features)

        if clean_batch and "pair_arm" in clean_batch[0]:
            raise ValueError(
                "Detected pair-orbit Sudoku samples (pair_arm). Use PairOrbitSudokuCollator and "
                "preprocess_sudoku_dataset(..., sudoku_pair_orbit_train=True)."
            )

        batch = super().__call__(clean_batch)
        batch["labels"] = batch["input_ids"].clone()
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        # `blank_token_mask` is only used to restrict which tokens are masked.
        # Remove it so it isn't passed to the model as an unexpected kwarg.
        batch.pop("blank_token_mask", None)
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)
            prompt_mask = prompt_length_indices < prompt_lengths
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()
        batch["input_ids"] = noisy_batch.long()
        return batch


class PairOrbitSudokuCollator(dLLMDataCollator):
    """
    Two D4 views per canonical puzzle share (t_b, canonical mask M_b). When refresh is enabled, each HF epoch can
    resample transforms and masking via ``pair_orbit_train_epoch`` (see ``PairOrbitEpochCallback``).
    """

    def __init__(
        self,
        *args,
        orbit_mask_eps: float = 1e-3,
        max_length: int = 512,
        prompt_style: str = "train_default",
        few_shot: bool = False,
        pair_orbit_refresh_each_epoch: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.orbit_mask_eps = float(orbit_mask_eps)
        self.pair_max_length = int(max_length)
        self.pair_prompt_style = str(prompt_style)
        self.pair_few_shot = bool(few_shot)
        self.pair_orbit_refresh_each_epoch = bool(pair_orbit_refresh_each_epoch)
        self.pair_orbit_train_epoch = 0

    def __call__(self, features):
        if not features or "pair_arm" not in features[0]:
            return super().__call__(features)

        # Contiguous (arm0, arm1) per puzzle is required below. If the DataLoader used a shuffled sampler,
        # reorder by pair_group_idx so both arms of each puzzle sit next to each other (when both are in batch).
        group_to_arms: dict[int, list[tuple[int, int]]] = defaultdict(list)
        group_order: list[int] = []
        for i, f in enumerate(features):
            gid = int(f["pair_group_idx"])
            if gid not in group_to_arms:
                group_order.append(gid)
            group_to_arms[gid].append((int(f["pair_arm"]), i))
        reordered: list[dict] = []
        for gid in group_order:
            arms = group_to_arms[gid]
            if len(arms) != 2:
                raise ValueError(
                    "PairOrbitSudokuCollator: each batch row must participate in a full pair (arm0 + arm1). "
                    f"group_id={gid} has {len(arms)} row(s) in this batch — training DataLoader is likely using "
                    "RandomSampler. Use dLLMPairOrbitTrainer (checks `pair_arm` on the dataset) so the train "
                    "loader uses SequentialSampler or PairOrbitDistributedSampler."
                )
            arms_sorted = sorted(arms, key=lambda x: x[0])
            if arms_sorted[0][0] != 0 or arms_sorted[1][0] != 1:
                raise ValueError(
                    f"PairOrbitSudokuCollator: group_id={gid} must have pair_arm 0 and 1; got {[a[0] for a in arms]}."
                )
            reordered.append(features[arms_sorted[0][1]])
            reordered.append(features[arms_sorted[1][1]])
        features = reordered

        n = len(features)
        if self.pair_orbit_refresh_each_epoch:
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                raise ValueError("PairOrbitSudokuCollator: tokenizer required when pair_orbit_refresh_each_epoch is True.")
            epoch_e = int(self.pair_orbit_train_epoch)
            prime = _PAIR_ORBIT_EPOCH_PRIME
            for srow in range(0, n, 2):
                erow = srow + 1
                if erow >= n:
                    break
                puzzle = features[srow].get("pair_canon_puzzle_flat")
                solution = features[srow].get("pair_canon_solution_flat")
                if puzzle is None or solution is None:
                    raise ValueError(
                        "pair_orbit_refresh_each_epoch requires each example to carry pair_canon_puzzle_flat and "
                        "pair_canon_solution_flat (re-run preprocessing with updated sft_trainer_pair)."
                    )
                mix = (int(features[srow]["pair_mask_seed"]) + epoch_e * prime) & 0x7FFFFFFF
                rng = random.Random(mix)
                tf_g, tf_h = rng.sample(range(len(ORBIT_TRANSFORMS)), 2)
                orig_idx = int(features[srow]["pair_group_idx"])
                pms = int(features[srow]["pair_mask_seed"])
                d0 = _pair_orbit_tokenize_one_arm(
                    puzzle,
                    solution,
                    tf_g,
                    tokenizer,
                    self.pair_max_length,
                    self.pair_prompt_style,
                    self.pair_few_shot,
                    orig_idx,
                    0,
                    pms,
                )
                d1 = _pair_orbit_tokenize_one_arm(
                    puzzle,
                    solution,
                    tf_h,
                    tokenizer,
                    self.pair_max_length,
                    self.pair_prompt_style,
                    self.pair_few_shot,
                    orig_idx,
                    1,
                    pms,
                )
                if d0 is None or d1 is None:
                    raise RuntimeError(
                        f"Pair-orbit per-epoch tokenization failed (orig_idx={orig_idx}); check grids and tokenizer."
                    )
                features[srow] = d0
                features[erow] = d1

        clusters: list[tuple[int, int]] = []
        pos = 0
        while pos < n:
            gid = int(features[pos]["pair_group_idx"])
            if int(features[pos]["pair_arm"]) != 0:
                raise ValueError("PairOrbitSudokuCollator: each puzzle block must start with pair_arm=0.")
            end = pos + 1
            if end >= n:
                raise ValueError("PairOrbitSudokuCollator: incomplete pair (missing arm 1).")
            if int(features[end]["pair_group_idx"]) != gid:
                raise ValueError("PairOrbitSudokuCollator: expected pair_group_idx match for two rows.")
            if int(features[end]["pair_arm"]) != 1:
                raise ValueError("PairOrbitSudokuCollator: second row must have pair_arm=1.")
            clusters.append((pos, end + 1))
            pos = end + 1

        device = features[0]["input_ids"].device if torch.is_tensor(features[0]["input_ids"]) else None
        input_ids = torch.stack([f["input_ids"] for f in features], dim=0).long()
        attn = torch.stack([f["attention_mask"] for f in features], dim=0).long()
        prompt_lengths = torch.stack([f["prompt_lengths"].view(-1)[0].long() for f in features], dim=0)
        assist_digit_pos = torch.stack([f["pair_assist_digit_pos"].long() for f in features], dim=0)
        canon_target = torch.stack([f["pair_canon_target_ids"].long() for f in features], dim=0)

        _, ncol = input_ids.shape
        clean_ids = input_ids.clone()
        mask_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        eps = max(self.orbit_mask_eps, 1e-6)
        supervise_rows: list[torch.Tensor] = []
        t_rows: list[torch.Tensor] = []

        perm_np = torch.from_numpy(_ORBIT_PERM_NP).long()
        cluster_id_assign = torch.zeros((n,), dtype=torch.long)

        for c_id, (s, e) in enumerate(clusters):
            for j in range(s, e):
                cluster_id_assign[j] = c_id

            seed_val = int(features[s]["pair_mask_seed"])
            if self.pair_orbit_refresh_each_epoch:
                seed_val = (seed_val + int(self.pair_orbit_train_epoch) * _PAIR_ORBIT_EPOCH_PRIME) % (2**31)
            cpu_gen = torch.Generator(device="cpu")
            cpu_gen.manual_seed(seed_val % (2**31))
            t_noise = (1.0 - eps) * torch.rand((1,), generator=cpu_gen).clamp(0.0, 1.0) + eps
            t_b = t_noise.squeeze(0).clamp(min=eps, max=1.0)
            rand81 = torch.rand(81, generator=cpu_gen)

            cb = features[s]["pair_canon_blank_cell"]
            if not torch.is_tensor(cb):
                cb = torch.tensor(cb, dtype=torch.bool)
            if device is None and torch.is_tensor(features[s]["input_ids"]):
                device = features[s]["input_ids"].device
            target_dev = device if device is not None else torch.device("cpu")
            cb = cb.to(target_dev).bool()
            rand81 = rand81.to(target_dev)
            t_b = t_b.to(target_dev)

            sup_c = cb & (rand81 < t_b)
            if not bool(sup_c.any()) and bool(cb.any()):
                blank_idxs = torch.nonzero(cb, as_tuple=False).squeeze(-1)
                nk = int(blank_idxs.numel())
                fb = torch.Generator(device="cpu")
                fb.manual_seed((seed_val + 917_411) % (2**31))
                j_pick = int(torch.randint(0, max(nk, 1), (1,), generator=fb).item())
                picked = blank_idxs[j_pick]
                sup_c = torch.zeros_like(cb)
                sup_c[picked] = True
            supervise_rows.append(sup_c)

            for j in range(s, e):
                g = int(features[j]["pair_view_idx"])
                perm = perm_np[g].to(target_dev)
                sup_v = sup_c[perm]
                dpos = assist_digit_pos[j].to(target_dev)
                for kcell in range(81):
                    if bool(sup_v[kcell]):
                        tp = int(dpos[kcell].item())
                        mask_indices[j, tp] = True
                t_rows.append(torch.full((ncol,), float(t_b.item()), device=target_dev))

        supervise_canon = torch.stack(supervise_rows, dim=0).float()
        t_expanded = torch.stack(t_rows, dim=0)

        noisy_batch = torch.where(mask_indices, int(self.mask_token_id), clean_ids.clone())
        labels = clean_ids.clone()
        labels[~mask_indices] = -100

        noisy_dev = noisy_batch.device
        seq = torch.arange(ncol, device=noisy_dev).unsqueeze(0)
        prompt_mask = seq < prompt_lengths.to(device=noisy_dev).unsqueeze(-1)
        noisy_batch[prompt_mask] = clean_ids[prompt_mask]
        labels[prompt_mask] = -100

        inv_perm = torch.from_numpy(_ORBIT_INV_NP).long()
        pair_view_batch = torch.tensor([int(features[j]["pair_view_idx"]) for j in range(n)], dtype=torch.long)

        return {
            "input_ids": noisy_batch.long(),
            "attention_mask": attn.to(device=noisy_dev),
            "labels": labels.long(),
            "t": t_expanded,
            "num_prompt_tokens": prompt_mask.sum(),
            "pair_B": torch.tensor(len(clusters), dtype=torch.long),
            "pair_supervise_canon": supervise_canon.to(device=noisy_dev),
            "pair_canon_target_ids": canon_target.to(device=noisy_dev),
            "pair_assist_digit_pos": assist_digit_pos.to(device=noisy_dev),
            "pair_inv_cell_perm": inv_perm,
            "pair_cluster_id": cluster_id_assign.to(device=noisy_dev),
            "pair_view_idx_batch": pair_view_batch.to(device=noisy_dev),
        }


# SYSTEM_PROMPT = """
# Respond in the following format:
# <reasoning>
# Your reasoning here
# </reasoning>
# <answer>
# ...
# </answer>
# """

# SYSTEM_PROMPT = """
# Please solve the following 9x9 Sudoku puzzle. The puzzle is provided as an 81-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

# Rules:
# - Fill empty cells with digits 1-9
# - Each row must contain digits 1-9 exactly once
# - Each column must contain digits 1-9 exactly once
# - Each 3x3 box must contain digits 1-9 exactly once

# Important: Your solution must be a COMPLETE 81-character string with only the digits 1-9, representing your final solved grid.
# """

SYSTEM_PROMPT = """
Please solve the following 9x9 Sudoku puzzle. The puzzle is provided as a 2D grid where digits are separated by spaces, rows are separated by newlines, and '.' represents empty cells.

Rules:
- Fill empty cells with digits 1-9
- Each row must contain digits 1-9 exactly once
- Each column must contain digits 1-9 exactly once
- Each 3x3 box must contain digits 1-9 exactly once

Important: Your solution must be a COMPLETE solved grid. You must maintain the exact same spatial format as the input: digits separated by spaces, with a newline after every row.
"""


def format_sudoku_grid(grid_text: str) -> str:
    grid_text = str(grid_text).strip()
    if not grid_text:
        return ""
    n = int(math.isqrt(len(grid_text)))
    if n * n != len(grid_text):
        raise ValueError(f"Sudoku grid must have square length, got {len(grid_text)} characters.")
    rows = []
    for row_idx in range(n):
        row = grid_text[row_idx * n : (row_idx + 1) * n]
        rows.append(" ".join("." if ch in {"0", "."} else ch for ch in row))
    return "\n".join(rows)


def _nine_by_nine_few_shot_example() -> str:
    return (
        "Worked example — a valid completed 9x9 grid (same rules; not your puzzle):\n"
        "1 2 3 4 5 6 7 8 9\n"
        "4 5 6 7 8 9 1 2 3\n"
        "7 8 9 1 2 3 4 5 6\n"
        "2 3 1 5 6 4 8 9 7\n"
        "5 6 4 8 9 7 2 3 1\n"
        "8 9 7 2 3 1 5 6 4\n"
        "3 1 2 6 4 5 9 7 8\n"
        "6 4 5 9 7 8 3 1 2\n"
        "9 7 8 3 1 2 6 4 5\n"
    )


def _infer_box_shape(n: int) -> tuple[int, int]:
    if n == 9:
        return 3, 3
    if n == 4:
        return 2, 2
    for r in range(int(math.sqrt(n)), 0, -1):
        if n % r == 0:
            return r, n // r
    return 1, n


def _build_rules_detailed_prompt(puzzle_text: str, n: int, few_shot: bool) -> str:
    box_rows, box_cols = _infer_box_shape(n)
    rules = (
        f"You are solving an {n}x{n} Sudoku.\n"
        f"- Use only digits 1 through {n} (each cell holds one digit).\n"
        f"- Each row must contain every digit from 1 to {n} exactly once.\n"
        f"- Each column must contain every digit from 1 to {n} exactly once.\n"
        f"- The board is tiled by non-overlapping {box_rows}x{box_cols} blocks "
        f"(there are {(n // box_rows) * (n // box_cols)} such blocks). "
        f"Each block must contain every digit from 1 to {n} exactly once.\n"
        "- The puzzle has exactly one valid solution consistent with the givens.\n"
    )
    footer = (
        "Return only the completed grid in the same format (spaces between digits, "
        "one row per line). Do not add explanations or extra text."
    )
    if few_shot and n == 9:
        rules = f"{rules}\n\n{_nine_by_nine_few_shot_example()}"
    return f"{rules}\nPuzzle:\n{puzzle_text}\n\n{footer}"


def build_sudoku_prompt(item, prompt_style: str = "train_default", few_shot: bool = False) -> str:
    puzzle = str(item["puzzle"]).strip()
    n = int(math.isqrt(len(puzzle)))
    if n * n != len(puzzle):
        raise ValueError(f"Expected a square Sudoku puzzle, got length {len(puzzle)}.")

    puzzle_text = format_sudoku_grid(puzzle)
    if prompt_style == "rules_detailed":
        return _build_rules_detailed_prompt(puzzle_text, n, few_shot)

    if few_shot and n == 9:
        return f"{SYSTEM_PROMPT}\n\n{_nine_by_nine_few_shot_example()}\n\nSolve the following Sudoku puzzle:\n{puzzle_text}\n"
    else:
        return f"{SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle:\n{puzzle_text}\n"


def normalize_grid_text(text: str) -> str:
    lines = []
    for raw_line in str(text).strip().splitlines():
        digits = [ch for ch in raw_line if ch.isdigit()]
        if digits:
            lines.append("".join(digits))
    return "\n".join(lines)


def canonical_sudoku_digits(text: str) -> str:
    """Row-major digit string for matching (ignores newlines between grid rows)."""
    return "".join(normalize_grid_text(text).splitlines())


def build_chat_example(tokenizer, prompt, response):
    messages = prompt + response
    inputs = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return inputs, prompt_text


def _preprocess_split(data, tokenizer, max_length, desc: str):
    """Tokenize chat examples.

    Skips rows with missing/None/empty first trajectory, and rows where ``max_length``
    truncation leaves no assistant tokens (prompt alone fills the non-padding span).
    """
    preprocessed_data = []
    for i in tqdm(range(len(data)), desc=desc):
        item = data[i]
        trajs = item.get("thinking_trajectories")
        attempt = item.get("attempt")
        if (
            trajs is None
            or not trajs
            or trajs[0] is None
            or trajs[0] == ""
            or attempt is None
        ):
            continue

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]

        # The dataset already includes structured reasoning/attempt strings;
        # we keep them concatenated to match the other SFT trainers.
        trajectory = f"{trajs[0]}{attempt}"

        response = [{"role": "assistant", "content": trajectory}]
        inputs, prompt_text = build_chat_example(tokenizer, prompt, response)
        tokenized_input = tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ).input_ids.squeeze(0)

        tokenized_prompt = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_length
        )
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            non_pad_len = int((tokenized_input != pad_id).sum().item())
        else:
            non_pad_len = int(tokenized_input.shape[0])
        start_search = int(tokenized_prompt.attention_mask.sum(-1).item())
        if start_search >= non_pad_len:
            continue

        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
                "orig_idx": i,
                "question": item["question"],
                "full_trajectory": trajectory,
            }
        )
    return preprocessed_data


def _preprocess_sudoku_split(
    data, tokenizer, max_length, desc: str, prompt_style: str = "train_default", few_shot: bool = False
):
    preprocessed_data = []
    for i in tqdm(range(len(data)), desc=desc):
        item = data[i]
        puzzle = item.get("puzzle")
        solution = item.get("solution")
        if puzzle is None or solution is None:
            continue

        puzzle = str(puzzle).strip()
        solution = str(solution).strip()
        if not puzzle or not solution:
            continue

        try:
            puzzle_text = format_sudoku_grid(puzzle)
            solution_text = format_sudoku_grid(solution)
            # puzzle_text = puzzle.replace(".", "0")
            # solution_text = solution.replace(".", "0")
        except ValueError:
            continue

        prompt = [{"role": "user", "content": build_sudoku_prompt(item, prompt_style=prompt_style, few_shot=few_shot)}]
        response = [{"role": "assistant", "content": solution_text}]
        inputs, prompt_text = build_chat_example(tokenizer, prompt, response)

        tokenized_input = tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ).input_ids.squeeze(0)
        tokenized_prompt = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_length
        )
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            non_pad_len = int((tokenized_input != pad_id).sum().item())
        else:
            non_pad_len = int(tokenized_input.shape[0])
        start_search = int(tokenized_prompt.attention_mask.sum(-1).item())
        if start_search >= non_pad_len:
            continue

        # Build a mask over token positions that correspond to *blank* cells
        # in the assistant grid. We detect digit tokens in the assistant portion
        # and align them with the puzzle cell order (row-major).
        # This matches eval behavior where only blanks are initialized as [MASK].
        try:
            n = int(math.isqrt(len(puzzle)))
            if n * n != len(puzzle):
                raise ValueError("puzzle length is not a perfect square")

            blank_chars = {".", "0", "_"}
            blank_cells = [idx for idx, ch in enumerate(puzzle) if ch in blank_chars]

            # Digit tokens (assume single-token digits; this is how eval does it too).
            digit_ids = []
            digit_id_set = set()
            for d in range(1, n + 1):
                ids = tokenizer(str(d), add_special_tokens=False).input_ids
                if len(ids) != 1:
                    raise ValueError(f"digit {d} does not map to a single token: {ids}")
                digit_ids.append(ids[0])
                digit_id_set.add(ids[0])

            blank_token_mask = torch.zeros((max_length,), dtype=torch.bool)
            # Scan only the assistant portion (tokens >= start_search).
            digit_positions_in_assistant = [
                pos
                for pos in range(start_search, max_length)
                if int(tokenized_input[pos].item()) in digit_id_set
            ]
            expected_digits = n * n
            if len(digit_positions_in_assistant) < expected_digits:
                raise ValueError(
                    f"Expected at least {expected_digits} digit tokens in assistant, got {len(digit_positions_in_assistant)}"
                )
            digit_positions_in_assistant = digit_positions_in_assistant[:expected_digits]

            for cell_idx in blank_cells:
                blank_token_mask[int(digit_positions_in_assistant[cell_idx])] = True

            # If puzzle has no blanks, there's nothing to train on (skip).
            if blank_token_mask.sum().item() == 0:
                continue
        except Exception:
            # Best-effort alignment; if the tokenizer layout is unexpected, skip the example.
            continue

        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
                "orig_idx": i,
                "prompt_text": prompt_text,
                "solution_text": solution_text,
                # Used by dLLMDataCollator to only mask blank cells.
                "blank_token_mask": blank_token_mask,
            }
        )
    return preprocessed_data


def _preprocess_sudoku_split_pair_orbit(
    data,
    tokenizer,
    max_length,
    desc: str,
    prompt_style: str = "train_default",
    few_shot: bool = False,
):
    """
    Each CSV row (canonical puzzle b): sample two distinct transforms g,h ∈ D4, emit two examples
    sharing ``pair_mask_seed`` (same t_b and canonical mask at collate). Collator may resample g,h and
    masking each HF epoch when ``pair_orbit_refresh_each_epoch`` is enabled.
    """
    groups: list[list[dict]] = []

    for i in tqdm(range(len(data)), desc=desc):
        item = data[i]
        puzzle = item.get("puzzle")
        solution = item.get("solution")
        if puzzle is None or solution is None:
            continue
        puzzle = str(puzzle).strip()
        solution = str(solution).strip()
        if not puzzle or not solution:
            continue
        pair_mask_seed = (int(i) * 1_000_003 + 0x9E3779B9) & 0x7FFFFFFF
        tf_g, tf_h = random.sample(range(len(ORBIT_TRANSFORMS)), 2)
        d0 = _pair_orbit_tokenize_one_arm(
            puzzle, solution, tf_g, tokenizer, max_length, prompt_style, few_shot, i, 0, pair_mask_seed
        )
        d1 = _pair_orbit_tokenize_one_arm(
            puzzle, solution, tf_h, tokenizer, max_length, prompt_style, few_shot, i, 1, pair_mask_seed
        )
        if d0 is None or d1 is None:
            continue
        groups.append([d0, d1])

    random.shuffle(groups)
    flat: list[dict] = []
    for grp in groups:
        flat.extend(grp)
    return flat


def preprocess_dataset(data, tokenizer, max_length, test_split=0.01, validation_data=None):
    """
    If `validation_data` is provided, preprocess train and validation splits separately.
    Otherwise, hold out `test_split` from the (shuffled) preprocessed training data.
    """
    if validation_data is not None:
        train_out = _preprocess_split(data, tokenizer, max_length, desc="Preprocessing train")
        val_out = _preprocess_split(validation_data, tokenizer, max_length, desc="Preprocessing validation")
        random.shuffle(train_out)
        return train_out, val_out

    preprocessed_data = _preprocess_split(data, tokenizer, max_length, desc="Preprocessing dataset")
    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data


def preprocess_sudoku_dataset(
    data,
    tokenizer,
    max_length,
    test_split=0.01,
    validation_data=None,
    prompt_style: str = "train_default",
    few_shot: bool = False,
    sudoku_pair_orbit_train: bool = False,
):
    if sudoku_pair_orbit_train and validation_data is None:
        raise ValueError(
            "preprocess_sudoku_dataset(..., sudoku_pair_orbit_train=True) requires validation_data=. "
            "Random train/val fractions would split pair blocks (every 2 rows must stay together)."
        )
    if validation_data is not None:
        if sudoku_pair_orbit_train:
            train_out = _preprocess_sudoku_split_pair_orbit(
                data,
                tokenizer,
                max_length,
                desc="Preprocessing Sudoku train (pair-orbit)",
                prompt_style=prompt_style,
                few_shot=few_shot,
            )
        else:
            train_out = _preprocess_sudoku_split(
                data,
                tokenizer,
                max_length,
                desc="Preprocessing Sudoku train",
                prompt_style=prompt_style,
                few_shot=few_shot,
            )
        val_out = _preprocess_sudoku_split(
            validation_data,
            tokenizer,
            max_length,
            desc="Preprocessing Sudoku validation",
            prompt_style=prompt_style,
            few_shot=few_shot,
        )
        if not sudoku_pair_orbit_train:
            random.shuffle(train_out)
        return train_out, val_out

    if sudoku_pair_orbit_train:
        raise ValueError("sudoku_pair_orbit_train=True requires explicit validation_data.")

    preprocessed_data = _preprocess_sudoku_split(
        data,
        tokenizer,
        max_length,
        desc="Preprocessing Sudoku dataset",
        prompt_style=prompt_style,
        few_shot=few_shot,
    )
    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data
