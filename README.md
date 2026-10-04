# SentimentScope

A concise BERT fine-tuning project for binary IMDB movie-review sentiment classification.

## Run

Install the dependencies, then open `SentimentScope.ipynb` and run the cells in order. The notebook uses `sentiment_scope.py` for the actual pipeline, which keeps training reusable from the command line:

```powershell
python sentiment_scope.py --epochs 2 --batch-size 16
```

For a quick smoke test, use `--train-samples 1000 --test-samples 500`. Use the full dataset and a CUDA GPU for the submission run.

## Generated outputs

The pipeline creates `artifacts/` during execution. Numeric results are deliberately persisted as CSV files:

- `data_summary.csv` — split, label, and review-length summary
- `training_metrics.csv` — loss and validation metrics per epoch
- `test_metrics.csv` — final loss, accuracy, precision, recall, and F1
- `test_predictions.csv` — prediction-level results
- `best_model/` — Hugging Face checkpoint and tokenizer

Generated figures are `review_lengths.png`, `training_curves.png`, and `confusion_matrix.png`.

If the installed PyTorch build does not support the available GPU, run a small verification job with `--device cpu`; use a compatible CUDA build for the full training run.
