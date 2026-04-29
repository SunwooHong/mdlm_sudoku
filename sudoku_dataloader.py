"""Sudoku 9x9 dataset + tokenizer for MDLM (length 81, vocab 10)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


class SudokuTokenizer:
  """Digits 1-9 -> ids 0-8, [MASK] -> 9. Kaggle puzzle blanks (0/.) map to [MASK].

  BOS/EOS/PAD are aliased to mask id so `Diffusion` / logging paths that expect
  these attributes do not crash; sequences are always length 81 (no padding).
  """

  def __init__(self) -> None:
    self.digits = '123456789'
    self.digit_to_id = {ch: i for i, ch in enumerate(self.digits)}
    self.id_to_digit = {i: ch for ch, i in self.digit_to_id.items()}

    self.mask_token = '[MASK]'
    self.mask_token_id = 9

    self.pad_token = '[MASK]'
    self.pad_token_id = self.mask_token_id
    self.bos_token = '[MASK]'
    self.bos_token_id = self.mask_token_id
    self.eos_token = '[MASK]'
    self.eos_token_id = self.mask_token_id
    self.unk_token = '[MASK]'
    self.unk_token_id = self.mask_token_id

    self.vocab_size = 10

  def encode(
      self,
      text: str,
      add_special_tokens: bool = False) -> List[int]:
    del add_special_tokens
    text = str(text).strip().replace(' ', '').replace('\n', '').replace('\t', '')
    ids: List[int] = []
    for ch in text:
      if ch in self.digit_to_id:
        ids.append(self.digit_to_id[ch])
      elif ch in ('0', '.', '_'):
        ids.append(self.mask_token_id)
      else:
        raise ValueError(f'Invalid Sudoku character: {ch!r}')
    return ids

  def decode(
      self,
      ids: Union[torch.Tensor, List[int], Any],
      skip_special_tokens: bool = False) -> str:
    del skip_special_tokens
    if isinstance(ids, torch.Tensor):
      ids = ids.detach().cpu().tolist()
    out: List[str] = []
    for idx in ids:
      idx = int(idx)
      if idx == self.mask_token_id:
        out.append('0')
      elif 0 <= idx <= 8:
        out.append(self.id_to_digit[idx])
      else:
        out.append('?')
    return ''.join(out)

  def batch_decode(
      self,
      batch: torch.Tensor,
      skip_special_tokens: bool = False) -> List[str]:
    if isinstance(batch, torch.Tensor):
      batch = batch.detach().cpu()
    return [self.decode(x, skip_special_tokens=skip_special_tokens) for x in batch]

  def get_vocab(self) -> Dict[str, int]:
    vocab = {ch: i for ch, i in self.digit_to_id.items()}
    vocab[self.mask_token] = self.mask_token_id
    return vocab


class SudokuNpyDataset(torch.utils.data.Dataset):
  """Memmaps produced by `scripts/prepare_sudoku9.py`."""

  def __init__(
      self,
      root: Union[str, Path],
      split: str,
      use_anchors: bool = False) -> None:
    root = Path(root)
    split = {
        'validation': 'valid',
        'val': 'valid',
        'valid': 'valid',
        'train': 'train',
        'test': 'test',
    }[split]

    self.solution_path = root / f'{split}_solution.npy'
    self.anchor_path = root / f'{split}_anchor.npy'
    self.use_anchors = use_anchors

    if not self.solution_path.exists():
      raise FileNotFoundError(f'Missing {self.solution_path}')

    self.solutions = np.load(self.solution_path, mmap_mode='r')

    if self.use_anchors:
      if not self.anchor_path.exists():
        raise FileNotFoundError(f'Missing {self.anchor_path}')
      self.anchors = np.load(self.anchor_path, mmap_mode='r')
    else:
      self.anchors = None

    assert self.solutions.ndim == 2 and self.solutions.shape[1] == 81
    if self.anchors is not None:
      assert self.anchors.shape == self.solutions.shape

  def __len__(self) -> int:
    return int(self.solutions.shape[0])

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor(self.solutions[idx].astype(np.int64), dtype=torch.long)

    if self.use_anchors:
      anchor_mask = torch.tensor(self.anchors[idx].astype(np.bool_), dtype=torch.bool)
      attention_mask = (~anchor_mask).long()
      return {
          'input_ids': input_ids,
          'attention_mask': attention_mask,
          'anchor_mask': anchor_mask,
      }

    return {
        'input_ids': input_ids,
        'attention_mask': torch.ones(81, dtype=torch.long),
    }


class SudokuCsvDataset(torch.utils.data.Dataset):
  """Sudoku dataset backed by split CSV files.

  Expected columns include `solution`; this loader uses only solution tokens for
  pretraining compatibility with existing Sudoku diffusion setup.
  """

  def __init__(self, root: Union[str, Path], split: str) -> None:
    root = Path(root)
    split = {
        'validation': 'val',
        'valid': 'val',
        'val': 'val',
        'train': 'pretraining',
        'test': 'test',
        'pretraining': 'pretraining',
        'post_training': 'post_training',
    }.get(split, split)
    csv_path = root / f'{split}.csv'
    if not csv_path.exists():
      raise FileNotFoundError(f'Missing {csv_path}')

    self.solutions: List[np.ndarray] = []
    with csv_path.open('r', newline='', encoding='utf-8') as f:
      reader = csv.DictReader(f)
      for row in reader:
        solution = str(row['solution']).strip()
        if len(solution) != 81:
          raise ValueError(f'Invalid solution length={len(solution)} in {csv_path}')
        ids = [int(ch) - 1 for ch in solution]
        if any(x < 0 or x > 8 for x in ids):
          raise ValueError(f'Invalid Sudoku solution digits in {csv_path}')
        self.solutions.append(np.asarray(ids, dtype=np.int64))

  def __len__(self) -> int:
    return len(self.solutions)

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor(self.solutions[idx], dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': torch.ones(81, dtype=torch.long),
    }
