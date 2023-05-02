# 483-583-project
Sentiment analysis with fine-tuned and zero-shot pre-trained large language models

# Running Few Shot
Run `sentiment_classifier_cv.py` with one command-line argument (float for `train_size`). Redirect output to file with corresponding name. Ex:

`python sentiment_classifier_cv.py 0.8 > few_shot_results/train_size_80.txt`

# Running Zero Shot
Run `zero_shot.ipynb` cells. Specify the pre-trained language model to use with `transformer_name` variable.
