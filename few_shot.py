import torch
import numpy as np
import pandas as pd
# set to True to use the gpu (if there is one available)
use_gpu = True

# select device
if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device =  -1
print("device", device)


# In[ ]:


from datasets import load_from_disk
dataset = load_from_disk("data/belief_dataset/")


# In[ ]:


from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')


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
batch_size = 4
num_epochs = 2
training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch",
                                  num_train_epochs=num_epochs,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size)

trainer = Trainer(
    model=nli_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
)


# In[ ]:


trainer.train()
print("\nfinish training")


# In[ ]:

print("\nzero shot classification pipeline")
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model=trainer.model, 
                      tokenizer=tokenizer,
                      device=device)


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
