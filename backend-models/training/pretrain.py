"""Train the domain-suggestion Transformer with next-token prediction.

Example:
    python -m training.pretrain --epochs 100 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from model.transformer import DomainTransformer, TransformerConfig
from training.dataset import NextTokenDataset, load_token_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the domain suggestion Transformer")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "cleaned" / "row.json")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "tokenizer" / "tokenizer.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "pretrain")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None, help="Defaults to sequence length")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
    return device


@torch.no_grad()
def evaluate(model: DomainTransformer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for input_ids, targets in loader:
        _, loss = model(input_ids.to(device), targets.to(device))
        losses.append(loss.item())
    return sum(losses) / len(losses)


def save_checkpoint(
    path: Path,
    model: DomainTransformer,
    optimizer: AdamW,
    epoch: int,
    train_loss: float,
    validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if not 0.0 <= args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be in [0, 0.5)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    token_ids = load_token_ids(args.data, args.tokenizer)
    dataset = NextTokenDataset(token_ids, args.sequence_length, args.stride)

    validation_count = max(1, round(len(dataset) * args.validation_fraction))
    if len(dataset) < 2:
        raise ValueError("Need at least two windows to create training and validation sets")
    validation_count = min(validation_count, len(dataset) - 1)
    train_count = len(dataset) - validation_count
    train_set, validation_set = random_split(
        dataset, [train_count, validation_count],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)

    config = TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_token_id=tokenizer.token_to_id("<pad>") or 0,
    )
    model = DomainTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_validation_loss = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8"
    )

    print(f"Device: {device}; tokens: {len(token_ids)}; windows: {len(dataset)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for input_ids, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids.to(device), targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        train_loss = sum(losses) / len(losses)
        validation_loss = evaluate(model, validation_loader, device)
        print(f"epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={validation_loss:.4f}")
        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, train_loss, validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, train_loss, validation_loss)


if __name__ == "__main__":
    main()
