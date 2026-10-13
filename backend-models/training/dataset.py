"""Dataset helpers for next-token pretraining on domain-suggestion records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tokenizers import Tokenizer


def record_to_text(record: dict) -> str:
    """Serialize a record exactly as it was serialized for tokenizer training."""
    return (
        f"Input: {record['input']}\n"
        f"Domain: {record['domain']}\n"
        f"Profession: {record['profession']}\n"
        f"Reason: {record['reason']}\n"
        f"Skills: {', '.join(record['skills'])}\n"
    )


def load_token_ids(
    data_path: str | Path,
    tokenizer_path: str | Path,
    bos_token: str = "<bos>",
    eos_token: str = "<eos>",
) -> list[int]:
    """Load records and turn them into one token stream separated by EOS tokens."""
    records = json.loads(Path(data_path).read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ValueError("Training data must be a non-empty JSON array")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos_id, eos_id = tokenizer.token_to_id(bos_token), tokenizer.token_to_id(eos_token)
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must contain <bos> and <eos> special tokens")

    token_ids: list[int] = []
    for record in records:
        required = {"input", "domain", "profession", "reason", "skills"}
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError("Every training record must use the expected five-field schema")
        token_ids.extend([bos_id, *tokenizer.encode(record_to_text(record)).ids, eos_id])
    return token_ids


class NextTokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """Fixed-length windows where targets are inputs shifted one token ahead."""

    def __init__(
        self,
        token_ids: Sequence[int],
        sequence_length: int,
        stride: int | None = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if len(token_ids) <= sequence_length:
            raise ValueError("Token stream must be longer than sequence_length")

        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        self.starts = list(range(0, len(self.tokens) - sequence_length, self.stride))
        if not self.starts:
            raise ValueError("No complete training windows were created")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = self.starts[index]
        window = self.tokens[start : start + self.sequence_length + 1]
        return window[:-1], window[1:]
