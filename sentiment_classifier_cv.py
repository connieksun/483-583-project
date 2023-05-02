from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel

import torch
torch.cuda.empty_cache()

import pandas as pd
import numpy as np
import os

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold

import sys
print ('Argument List:', str(sys.argv))
train_size = float(sys.argv[1])
print()

models_dir = "models/bert-k-fold"

labels = open('data/classes.txt').read().splitlines()
df = pd.read_csv("data/belief_benchmark_all_train.csv")
df.head()

transformer_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(transformer_name)
# NOTE: for cross validation, the model should be initialized inside the cv loop

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True)

def compute_metrics(eval_pred):
    y_true = eval_pred.label_ids
    y_pred = np.argmax(eval_pred.predictions, axis=-1)
    report = metrics.classification_report(y_true, y_pred)
    print("report: \n", report)
    
    print("rep type: ", type(report))
    

    return {'f1':metrics.f1_score(y_true, y_pred, average="macro")}

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        cls_outputs = outputs.last_hidden_state[:, 0, :]
        cls_outputs = self.dropout(cls_outputs)
        logits = self.classifier(cls_outputs)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# this is for creating cross-validation folds
def get_sample_based_on_idx(data, indeces):
    return data.iloc[indeces, :].reset_index()

# defining hyperparams
num_epochs = 8
batch_size = 6
weight_decay = 0.01
print(f"num_epochs: {num_epochs}, batch_size: {batch_size}, weight_decay: {weight_decay}")
training_args = TrainingArguments(
    output_dir="./results_sentiment_analysis",
    log_level='error',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    weight_decay=weight_decay,
    load_best_model_at_end=True, # this is supposed to make sure the best model is loaded by the trainer at the end
    metric_for_best_model="eval_f1" 
    )

# specify what percent of the all_train data should be used in cross-fold validation 
# we can use this to create a learning curve, e.g., performance with 50, 80, 90% of data
if train_size < 1.0:
    df, _ = train_test_split(df, test_size=1.0-train_size, random_state=1, stratify=df[['label']])
print(f"train_size: {train_size} (total samples: {len(df)})\n")

output = open("original.txt", "a")

fold = 0
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
for train_df_idx, eval_df_idx in kfold.split(df):
    
    print(f"************** BEGIN FOLD: {fold+1} **************")
    output.write(f"FOLD: {fold}\n")
    new_df = pd.DataFrame()
    
    train_df = get_sample_based_on_idx(df, train_df_idx)
    print("LEN DF: ", len(train_df))
    output.write(f"LEN DF: {len(train_df)}\n")

    print("done train df")
    output.write("done train df\n")
    eval_df = get_sample_based_on_idx(df, eval_df_idx)

    print("done eval df")
    output.write("done eval df\n")
    print("LEN EVAL: ", len(eval_df))
    output.write(f"LEN EVAL: {len(eval_df)}\n")

    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train_df)
    ds['validation'] = Dataset.from_pandas(eval_df)
    train_ds = ds['train'].map(
        tokenize, batched=True
    )
    eval_ds = ds['validation'].map(
        tokenize,
        batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(transformer_name, num_labels=4)
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    # after training, predict (will use best model?)
    preds = trainer.predict(eval_ds)
#     print("HERE: " , preds)
    final_preds = [np.argmax(x) for x in preds.predictions]
    real_f1 = metrics.f1_score(final_preds, eval_df["label"], average="macro")
    print("F-1: ", real_f1)
    output.write(f"F-1: {real_f1}\n")
    model_name = f"{transformer_name}-best-of-fold-{fold}-f1-{real_f1}"
    model_dir = os.path.join(models_dir, model_name)

    trainer.save_model(model_dir)

    for i, item in enumerate(final_preds):
        if item != eval_ds["label"][i]:
            wrong_df = pd.DataFrame()
            wrong_df["text"] = [eval_df["text"][i]]
            wrong_df["real"] = [eval_df["label"][i]]
            wrong_df["predicted"] = [item]
            new_df = pd.concat([new_df, wrong_df])

    new_df.to_csv(f"{models_dir}/wrong_predictions_{fold}.csv")

    print(f"************** END FOLD: {fold+1} **************\n")
    fold += 1


print("\n******************* holdout results ******************* ")
holdout_df = df = pd.read_csv("data/belief_benchmark_holdout.csv")
holdout_ds = Dataset.from_pandas(holdout_df)
holdout_ds = holdout_ds.map(tokenize, batched=True)

preds = trainer.predict(holdout_ds)

final_preds = [np.argmax(x) for x in preds.predictions]
real_f1 = metrics.f1_score(final_preds, holdout_df["label"], average="macro")
print("F-1: ", real_f1)
y_pred = []
for i, item in enumerate(final_preds):
    y_pred.append(item)

y_true = holdout_ds["label"]

print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
import pickle
pickle.dump(y_pred, open(f"few_shot_results/y_pred_{int(train_size*100)}", "wb"))
pickle.dump(y_true, open(f"few_shot_results/y_true_{int(train_size*100)}", "wb"))

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
output.write("{torch.cuda.memory_summary(device=None, abbreviated=False)}")

output.close()