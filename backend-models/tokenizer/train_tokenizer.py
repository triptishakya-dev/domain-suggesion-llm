import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "row.json"
OUTPUT_DIR = PROJECT_ROOT / "tokenizer"

VOCAB_SIZE = 8192
MIN_FREQUENCY = 1


def record_to_text(record):
    skills = ", ".join(record["skills"])

    return (
        f"Input: {record['input']}\n"
        f"Domain: {record['domain']}\n"
        f"Profession: {record['profession']}\n"
        f"Reason: {record['reason']}\n"
        f"Skills: {skills}\n"
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError("Input dataset must be a non-empty JSON array")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_file = OUTPUT_DIR / "tokenizer_training.txt"

    with training_file.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(record_to_text(record))
            file.write("\n")

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<unk>"
        )
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=[
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
        ],
        show_progress=True,
    )

    tokenizer.train(
        files=[str(training_file)],
        trainer=trainer,
    )

    tokenizer_json = OUTPUT_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))

    vocab = tokenizer.get_vocab()
    vocab_sorted = dict(
        sorted(vocab.items(), key=lambda item: item[1])
    )

    with (OUTPUT_DIR / "vocab.json").open("w", encoding="utf-8") as file:
        json.dump(vocab_sorted, file, ensure_ascii=False, indent=2)

    report = {
        "input_file": str(INPUT_FILE),
        "record_count": len(records),
        "vocab_size_requested": VOCAB_SIZE,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "special_tokens": [
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
        ],
        "tokenizer_type": "ByteLevel BPE",
    }

    with (OUTPUT_DIR / "tokenizer_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("Tokenizer training completed.")
    print(f"Records: {len(records)}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Saved to: {tokenizer_json}")


if __name__ == "__main__":
    main()