from __future__ import annotations

from pathlib import Path
from typing import Literal

import lightning as L
import torch


class PhaseSplitCheckpoint(L.pytorch.callbacks.Callback):
  """Save main/later best and last checkpoints split by global step boundary."""

  def __init__(
      self,
      boundary_step: int,
      monitor: str = 'val/nll',
      mode: Literal['min', 'max'] = 'min',
      dirpath: str = 'checkpoints') -> None:
    super().__init__()
    if boundary_step <= 0:
      raise ValueError(f'boundary_step must be positive, got {boundary_step}')
    if mode not in ('min', 'max'):
      raise ValueError(f'mode must be min/max, got {mode}')
    self.boundary_step = int(boundary_step)
    self.monitor = monitor
    self.mode = mode
    self.dirpath = Path(dirpath)

    self.main_dir = self.dirpath / 'main'
    self.later_dir = self.dirpath / 'later'
    self.best_main: float | None = None
    self.best_later: float | None = None
    self._main_last_saved = False

  def _is_better(self, cur: float, prev: float | None) -> bool:
    if prev is None:
      return True
    if self.mode == 'min':
      return cur < prev
    return cur > prev

  def _save(self, trainer: L.Trainer, path: Path) -> None:
    # During sanity validation, train dataloader is not initialized yet.
    # Diffusion.on_save_checkpoint expects train_dataloader.sampler, so skip.
    train_dl = getattr(trainer, 'train_dataloader', None)
    if train_dl is None:
      return
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(path))

  def _get_metric(self, trainer: L.Trainer) -> float | None:
    value = trainer.callback_metrics.get(self.monitor)
    if value is None:
      return None
    if isinstance(value, torch.Tensor):
      value = value.detach().float().cpu().item()
    return float(value)

  def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
    del pl_module
    if getattr(trainer, 'sanity_checking', False):
      return
    metric = self._get_metric(trainer)
    if metric is None:
      return
    step = int(trainer.global_step)
    if step <= self.boundary_step:
      if self._is_better(metric, self.best_main):
        self.best_main = metric
        self._save(trainer, self.main_dir / 'best.ckpt')
    else:
      if self._is_better(metric, self.best_later):
        self.best_later = metric
        self._save(trainer, self.later_dir / 'best_later.ckpt')

  def on_train_batch_end(
      self,
      trainer: L.Trainer,
      pl_module: L.LightningModule,
      outputs,
      batch,
      batch_idx: int) -> None:
    del pl_module, outputs, batch, batch_idx
    step = int(trainer.global_step)
    if not self._main_last_saved and step >= self.boundary_step:
      self._save(trainer, self.main_dir / 'last.ckpt')
      self._main_last_saved = True

  def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
    del pl_module
    step = int(trainer.global_step)
    if step <= self.boundary_step and not self._main_last_saved:
      self._save(trainer, self.main_dir / 'last.ckpt')
      self._main_last_saved = True
    if step > self.boundary_step:
      self._save(trainer, self.later_dir / 'last_later.ckpt')
