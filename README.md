# Sentiment Analysis for Belief Extraction
Authors: Rohini Saran, Caroline Hyland, Luke Hankins, and Connie Sun

CSC 483/583, Spring 2023, The University of Arizona

# Overview
Sentiment analysis with zero-shot and fine-tuned pre-trained large language models on Senegal River Valley dataset.

This project uses the pre-trained BERT language model to predict whether statements have a sentiment that is positive, negative, both, or neither. We implemented both a zero-shot classification pipeline that takes no training data and a fine-tuning method that trains BERT for sequence classification on varying training dataset sizes. Other LMs can be evaluated with minimal modifications to the code.

# How to Use
## Running Zero Shot
Run all `zero_shot.ipynb` cells. Specify the pre-trained language model to use with `transformer_name` variable. Specify the dataset to predict on (either all data or holdout set) in the `for` loop of cell 8, line 9. Outputs a classification report with precision, recall, f1-scores, and accuracy. Also pickles the true and predicted label lists in the `few_shot_results/` directory.

## Running Fine-Tuning
Run all `sentiment_classifier_cv.ipynb` cells. Specify the hyperparameters in cell 12, lines 2-4 (`num_epochs`, `batch_size`, `weight_decay`). Specify the proportion of training data to use in cross-fold validation (training) step, cell 13 line 3.

Alternatively, run the python script version `sentiment_classifier_cv.py`. Takes one command-line argument, a float value for `train_size`. Redirect the output to a file with corresponding name. Ex:

`python sentiment_classifier_cv.py 0.8 > few_shot_results/train_size_80.txt`

Both scripts will output classification reports for each epoch of each fold, and a final classification report for performance on the holdout set. The true and predicted labels are pickled and saved to the `few_shot_results/` directory.

*Note: Current scripts use the same seed for sampling data, so all results in `few_shot_results/` should be reproducible.

# Included Files
- `archived/*` -- Contains old scripts and results created throughout the course of this project that are no longer relevant to our main implementation and results.
- `data/*` -- Contains raw data, processed data, and static training/holdout datasets.
- `few_shot_results/*` -- Results from running `sentiment_classifier_cv.*` on various training dataset sizes. Also includes `zero_shot.ipynb` results on bert-base-cased model. Includes output .txt files with classification reports and pickled prediction and true label lists. 
- `.gitignore` -- Exclude model checkpoint folders
- `README.md` -- This file
- `data_processing.ipynb` -- File for processing Senegal River Valley dataset from raw csv form. Outputs the processed data with sentiment labels and generates static training and holdout datasets. 
- `data_visualization.ipynb` -- File for visualizing predicted vs. true values. Loads saved lists of predicted and true values and outputs classification reports and confusion matrices.
- `original.txt` -- Output file created when running fine-tuning; saves the evaluation results during training.
- `sentiment_classifier_cv.ipynb` -- Main file for running fine-tuning on pre-trained BERT model. Runs k-fold cross validation on a training dataset and evaluates fine-tuned model on a holdout set.
- `sentiment_classifier_cv.py` -- Python script version of fine-tuning code. Takes command-line argument for training dataset size.
- `zero_shot.ipynb` -- Main file for running zero-shot classification on pre-trained language model. Evaluates on both entire dataset and holdout set.
