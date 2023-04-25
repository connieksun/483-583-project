# 483-583-project
Belief extraction and semantic analysis

# Running Few Shot
Run `sentiment_classifier_cv.py` with one command-line argument (float for `train_size`). Redirect output to file with corresponding name. Ex:

`python sentiment_classifier_cv.py 0.8 > few_shot_results/train_size_80.txt`

# Annotating Data
Everyone will look at 150 statements from the fuzzy beliefs dataset and verify the person below
1. first round done by Saturday 3/18
2. second round done by Wednesday 3/22
- sentiment: [positive, negative, undetermined]
- topic: [chemicals, machinery, policy, economy, climate, ...]

# Create Environment
Test that environment works by running `imdb_test.ipynb`

# Sentihood Test
https://github.com/yhcc/BARTABSA

`conda env create --name transformers --file=transformers-env.yml`