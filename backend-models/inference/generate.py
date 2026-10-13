"""Load the trained domain Transformer and generate a suggestion."""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.transformer import DomainTransformer, TransformerConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "pretrain" / "best.pt"
DEFAULT_TOKENIZER = PROJECT_ROOT / "tokenizer" / "tokenizer.json"


class DomainSuggestionGenerator:
    """A reusable, inference-only wrapper around the saved checkpoint."""

    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT, tokenizer_path: str | Path = DEFAULT_TOKENIZER, device: str | torch.device = "auto") -> None:
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        checkpoint_path, tokenizer_path = Path(checkpoint_path), Path(tokenizer_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = DomainTransformer(TransformerConfig(**checkpoint["model_config"]))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.bos_token_id = self._token_id("<bos>")
        self.eos_token_id = self._token_id("<eos>")

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required token {token}")
        return token_id

    @torch.inference_mode()
    def suggest(self, message: str, max_new_tokens: int = 96, temperature: float = 0.7, top_k: int = 25) -> str:
        message = message.strip()
        if not message:
            raise ValueError("message must not be empty")
        if len(message) > 1_000:
            raise ValueError("message must be 1,000 characters or fewer")
        prompt = f"Input: {message}\nDomain:"
        prompt_ids = [self.bos_token_id, *self.tokenizer.encode(prompt).ids]
        prompt_ids = prompt_ids[-self.model.config.max_sequence_length :]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated = self.model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, eos_token_id=self.eos_token_id)[0].tolist()
        completion = self.tokenizer.decode(generated[len(prompt_ids):], skip_special_tokens=True).strip()
        return f"Domain: {completion}" if completion else "I could not generate a suggestion. Please try again."
