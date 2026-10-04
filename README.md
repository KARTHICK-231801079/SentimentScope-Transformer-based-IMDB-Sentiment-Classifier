# SentimentScope

Train a small transformer classifier from scratch to label IMDB movie reviews as positive or negative. The notebook uses the `bert-base-uncased` tokenizer and a custom transformer with mean pooling and a classification head.

## Project files

- `SentimentScope.ipynb` — data exploration, model training, evaluation, and prediction.
- `aclImdb/` — IMDB reviews, organized into `train/pos`, `train/neg`, `test/pos`, and `test/neg`.

## Setup

Use Python 3.10 or newer and install the notebook dependencies in your project environment:

```powershell
python -m pip install torch pandas matplotlib transformers
```

Open `SentimentScope.ipynb` in Jupyter or VS Code and select that environment as the notebook kernel. On first run, the Hugging Face tokenizer may need an internet connection to download its files.

## Run

Run the notebook cells from top to bottom. The notebook loads reviews from `aclImdb/`, creates training and validation splits, trains for five epochs, and reports validation and test accuracy. It currently forces training onto the CPU for compatibility with the configured PyTorch build. CPU training may take a while.

To use a GPU, install a PyTorch build compatible with your GPU and CUDA version, then update the training cell's `device` setting. The installed build must support the GPU's compute capability; otherwise CUDA operations can fail even when `torch.cuda.is_available()` is `True`.

## Generated outputs

The notebook writes these files to its current working directory:

- `sentimentscope.pt` — trained model weights.
- `sentiment_predictions.csv` — example predictions with review text, sentiment, and confidence.
- `label_distribution.png` — counts of positive and negative training reviews.
- `review_length_distribution.png` — histogram of training review lengths.

The `predict(texts, model, tokenizer, device)` function can also classify a list of custom reviews. Edit `prediction_texts` in the final notebook cell to choose which reviews to export to CSV.
