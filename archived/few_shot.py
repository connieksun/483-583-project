import torch
import numpy as np
import pandas as pd
# set to True to use the gpu (if there is one available)
use_gpu = True

# select device
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
print(f'device: {device.type}')


# In[ ]:


import pandas as pd
labels = open('data/classes.txt').read().splitlines()
all_df = pd.read_csv("data/belief_benchmark.csv")
from datasets import Dataset

dataset = Dataset.from_pandas(all_df)


# In[ ]:


from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


# In[ ]:


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


# In[ ]:


data_dict = tokenized_datasets.train_test_split(test_size=0.95)
train_ds = data_dict['train']

all_test_ds = data_dict['test']
all_test_dict = all_test_ds.train_test_split(test_size=0.80)
eval_ds = all_test_dict['train']
test_ds = all_test_dict['test']


for entry in train_ds:
    print(entry['label'])


# In[ ]:


from transformers import TrainingArguments
training_args = TrainingArguments(output_dir="models/few-shot-bart-large-mnli")

import numpy as np
import evaluate
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


from transformers import TrainingArguments, Trainer
batch_size = 6
num_epochs = 2
weight_decay = .01
training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch",
                                  num_train_epochs=num_epochs,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=weight_decay)

trainer = Trainer(
    model=nli_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
)


# In[ ]:

print("\nfew shot training")
torch.cuda.empty_cache()
trainer.train()
print("\nfinish training")


# In[ ]:

print("\nzero shot classification pipeline")
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model=trainer.model, 
                      tokenizer=tokenizer)


# In[ ]:


from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

labels = ["neither", "negative", "positive", "both"]
pred_i_list = []
for pred in tqdm(classifier(KeyDataset(test_ds, "text"), batch_size=4, candidate_labels=labels)):
    pred_i = labels.index(pred["labels"][0])
    pred_i_list.append(pred_i)


# In[ ]:


from sklearn.metrics import classification_report

y_true = test_ds["label"]
y_pred = pred_i_list
print()
print(classification_report(y_true, y_pred, target_names=labels))

