# Model Report - DistilBERT Guardrail Classifier

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `distilbert-base-uncased` |
| Training data | `tests/fixtures/training_data.jsonl` |
| Epochs | 4 |
| Batch size | 8 |
| Train / val split | 80 / 20 |

## Validation Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.9828 |
| Precision (malicious) | 1.0000 |
| Recall (malicious) | 0.9667 |
| F1 (malicious) | 0.9831 |
| ROC AUC | 0.9976 |
| False-positive rate (benign→block) | 0.0% |

## Confusion Matrix

Rows = actual class, columns = predicted class.

| | Predicted benign | Predicted malicious |
|---|---|---|
| Actual benign | 28 | 0 |
| Actual malicious | 1 | 29 |

## Quality Thresholds

| Criterion | Result | Pass? |
|-----------|--------|-------|
| Validation accuracy >= 90% | 98.3% | pass |
| F1 score documented | 0.9831 | pass |
| False-positive rate < 5% | 0.0% | pass |
