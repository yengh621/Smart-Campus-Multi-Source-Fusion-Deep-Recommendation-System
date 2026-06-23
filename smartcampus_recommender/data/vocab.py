from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Vocabulary:
    """0=PAD，1=UNK，其余为有效类别。"""
    token_to_idx: dict[str, int] = field(default_factory=lambda: {"<PAD>": 0, "<UNK>": 1})
    idx_to_token: list[str] = field(default_factory=lambda: ["<PAD>", "<UNK>"])

    def add(self, token: object) -> int:
        key = str(token)
        if key not in self.token_to_idx:
            self.token_to_idx[key] = len(self.idx_to_token)
            self.idx_to_token.append(key)
        return self.token_to_idx[key]

    def build(self, tokens: Iterable[object]) -> "Vocabulary":
        for token in tokens:
            self.add(token)
        return self

    def encode(self, token: object) -> int:
        return self.token_to_idx.get(str(token), 1)

    def decode(self, index: int) -> str:
        return self.idx_to_token[index] if 0 <= index < len(self.idx_to_token) else "<UNK>"

    def __len__(self) -> int:
        return len(self.idx_to_token)

