# SentimentScope Project Results

## Project Summary

The project uses the IMDB movie-review dataset, loading 25,000 labeled training reviews and 25,000 test reviews from the positive and negative folders. The data is explored by checking the label distribution, review lengths, and sample reviews. The training set is shuffled and split into 22,500 training reviews and 2,500 validation reviews.

A custom PyTorch dataset tokenizes each review with the `bert-base-uncased` tokenizer and pads or truncates each sequence to 128 tokens. A transformer classifier built from scratch is trained on CPU for five epochs. Validation accuracy rises from 73.36% after the first epoch to 81.56% after the fifth. The final recorded test accuracy is 78.51%. The model weights are saved to `sentimentscope.pt`, and example predictions are exported to `sentiment_predictions.csv`.

## Key Takeaways

- The training and test sets each contain 25,000 reviews, with positive and negative reviews represented equally. This provides a balanced binary classification task.
- Validation accuracy improves across training, but test accuracy is 3.05 percentage points lower than the final validation accuracy. This indicates a modest generalization gap and room for improvement on unseen reviews.
- The model processes at most 128 tokens per review, so sentiment information beyond that limit may be missed in longer reviews.
