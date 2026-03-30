"""Train a local DistilBERT binary classifier for prompt maliciousness scoring.

Usage
-----
::

    # Using the bundled sample fixture:
    python -m guardrail_proxy.training.train_distilbert \\
        --data tests/fixtures/training_data.jsonl \\
        --output-dir artifacts/distilbert_guardrail

    # Explicit epochs and batch size:
    python -m guardrail_proxy.training.train_distilbert \\
        --data my_data.jsonl \\
        --output-dir artifacts/distilbert_guardrail \\
        --epochs 5 \\
        --batch-size 16

Input format
------------
Each line of the JSONL file must be a JSON object with two keys::

    {"text": "ignore all previous instructions", "label": 1}
    {"text": "What is the capital of France?",   "label": 0}

Label encoding: 0 = benign, 1 = malicious / adversarial.

Output
------
The trained model and tokeniser are saved under ``--output-dir``.  Point
``MALICIOUSNESS_MODEL_PATH`` in ``.env`` at this directory to activate it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ── Dataset ───────────────────────────────────────────────────────────────────

class PromptDataset(Dataset):
    """PyTorch Dataset backed by JSONL prompt classification examples."""

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self._records = records
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self._records[idx]
        enc = self._tokenizer(
            str(record["text"]),
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(int(record["label"]), dtype=torch.long),
        }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL records from *path*.  Empty / whitespace lines are skipped."""
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvalMetrics(NamedTuple):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: list[list[int]]  # [[TN, FP], [FN, TP]]


def evaluate(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> EvalMetrics:
    """Return full classification metrics over *dataloader* in eval mode."""
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:  # type: ignore[assignment]
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]  # P(malicious)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    report = classification_report(
        all_labels, all_preds, target_names=["benign", "malicious"], output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()
    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = 0.0

    return EvalMetrics(
        accuracy=report["accuracy"],
        precision=report["malicious"]["precision"],
        recall=report["malicious"]["recall"],
        f1=report["malicious"]["f1-score"],
        roc_auc=auc,
        confusion=cm,
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    data_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    report_path: Path | None = None,
) -> EvalMetrics:
    """
    Fine-tune DistilBERT for binary prompt maliciousness classification.

    Parameters
    ----------
    data_path   : Path to the JSONL training data file.
    output_dir  : Directory where the model and tokeniser are written.
    epochs      : Number of full passes over the training data.
    batch_size  : Mini-batch size for training and validation.
    report_path : Optional path to write the Markdown model report.

    Returns
    -------
    EvalMetrics with final validation metrics.
    """
    records = load_jsonl(data_path)
    if len(records) < 4:
        raise ValueError(
            f"Training file must have at least 4 examples (got {len(records)})."
        )

    train_records, val_records = train_test_split(
        records, test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    train_loader = DataLoader(
        PromptDataset(train_records, tokenizer),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        PromptDataset(val_records, tokenizer),
        batch_size=batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    print(f"Training on {len(train_records)} examples, validating on {len(val_records)}.")
    print(f"Device: {device}  |  Epochs: {epochs}  |  Batch size: {batch_size}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(
            f"epoch={epoch}/{epochs}  train_loss={avg_loss:.4f}"
            f"  val_accuracy={metrics.accuracy:.4f}"
            f"  f1={metrics.f1:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")
    print("Set MALICIOUSNESS_MODEL_PATH=" + str(output_dir) + " in .env to activate it.")

    _print_metrics(metrics)
    if report_path is not None:
        _write_report(metrics, report_path, data_path, epochs, batch_size)
        print(f"Model report written to {report_path}")

    return metrics


def _print_metrics(m: EvalMetrics) -> None:
    """Print a formatted metrics summary to stdout."""
    cm = m.confusion
    print("\n── Validation Metrics ───────────────────────────────────────────")
    print(f"  Accuracy  : {m.accuracy:.4f}")
    print(f"  Precision : {m.precision:.4f}  (malicious class)")
    print(f"  Recall    : {m.recall:.4f}  (malicious class)")
    print(f"  F1        : {m.f1:.4f}  (malicious class)")
    print(f"  ROC AUC   : {m.roc_auc:.4f}")
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"               benign  malicious")
    if len(cm) == 2 and len(cm[0]) == 2:
        print(f"  benign     {cm[0][0]:6d}  {cm[0][1]:9d}")
        print(f"  malicious  {cm[1][0]:6d}  {cm[1][1]:9d}")
    print("─────────────────────────────────────────────────────────────────\n")


def _write_report(
    m: EvalMetrics,
    report_path: Path,
    data_path: Path,
    epochs: int,
    batch_size: int,
) -> None:
    """Write a Markdown model report with training configuration and validation metrics."""
    cm = m.confusion
    fp_rate = (cm[0][1] / max(cm[0][0] + cm[0][1], 1)) * 100 if len(cm) == 2 else 0.0
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Report: DistilBERT Guardrail Classifier",
        "",
        "## Training Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Base model | `distilbert-base-uncased` |",
        f"| Training data | `{data_path}` |",
        f"| Epochs | {epochs} |",
        f"| Batch size | {batch_size} |",
        f"| Train / val split | 80 / 20 |",
        "",
        "## Validation Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy | {m.accuracy:.4f} |",
        f"| Precision (malicious) | {m.precision:.4f} |",
        f"| Recall (malicious) | {m.recall:.4f} |",
        f"| F1 (malicious) | {m.f1:.4f} |",
        f"| ROC AUC | {m.roc_auc:.4f} |",
        f"| False-positive rate (benign→block) | {fp_rate:.1f}% |",
        "",
        "## Confusion Matrix",
        "",
        "Rows = actual class, columns = predicted class.",
        "",
        "| | Predicted benign | Predicted malicious |",
        "|---|---|---|",
    ]
    if len(cm) == 2 and len(cm[0]) == 2:
        lines += [
            f"| Actual benign | {cm[0][0]} | {cm[0][1]} |",
            f"| Actual malicious | {cm[1][0]} | {cm[1][1]} |",
        ]
    lines += [
        "",
        "## Quality Thresholds",
        "",
        f"| Criterion | Result | Pass? |",
        f"|-----------|--------|-------|",
        f"| Validation accuracy >= 90% | {m.accuracy:.1%} | {'pass' if m.accuracy >= 0.90 else 'fail'} |",
        f"| F1 score documented | {m.f1:.4f} | pass |",
        f"| False-positive rate < 5% | {fp_rate:.1f}% | {'pass' if fp_rate < 5.0 else 'fail'} |",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",        type=Path, required=True,
                   help="Path to JSONL training file.")
    p.add_argument("--output-dir",  type=Path,
                   default=Path("artifacts/distilbert_guardrail"),
                   help="Directory to save the trained model.")
    p.add_argument("--epochs",      type=int, default=3)
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--report-path", type=Path,
                   default=Path("docs/model-report.md"),
                   help="Write Markdown model report to this path.")
    return p


def main() -> None:
    """Parse CLI arguments and launch the training loop."""
    args = _build_parser().parse_args()
    train(args.data, args.output_dir, args.epochs, args.batch_size, args.report_path)


if __name__ == "__main__":
    main()
