import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


with open('kor_review.pk', 'rb') as f:
    data = pickle.load(f)
    
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

MODEL_NAME = 'beomi/KcELECTRA-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenized_train_sentence = tokenizer(
    list(train_data['review']),
    return_tensors='pt',
    max_length=256,
    padding =True,
    truncation=True,
    add_special_tokens=True,
)

tokenized_test_sentence = tokenizer(
    list(test_data['review']),
    return_tensors='pt',
    max_length=256,
    padding =True,
    truncation=True, # maxlen보다 더 긴 문장 들어오면 자름
    add_special_tokens=True, # 토큰 id 추가
)

class CurseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
train_label = train_data['label'].values
test_label = test_data['label'].values

train_dataset = CurseDataset(tokenized_train_sentence, train_label)
test_dataset = CurseDataset(tokenized_test_sentence, test_label)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = 2) 
# 실험을 해보자!! cross entropy  / num_labels 2~3 실패 분석!!
model.to(device)

training_args = TrainingArguments(
    output_dir='./result',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_dir='./logs',
    logging_steps=500,
    save_steps = 10000,
)

def metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy' : acc,
        'f1' : f1,
        'precision' : precision,
        'recall' : recall
    }
    
    
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    compute_metrics = metrics
)

trainer.train()
torch.save(model.state_dict(), "./kor_model.pth")
trainer.evaluate(eval_dataset=test_dataset)

