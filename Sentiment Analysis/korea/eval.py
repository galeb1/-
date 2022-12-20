import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import ElectraForSequenceClassification, ElectraTokenizer, AdamW

MODEL_NAME = 'beomi/KcELECTRA-base'
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = 2)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.load_state_dict(torch.load("./kor_model.pth",map_location=torch.device('cpu')))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def sentence_predict(sent):
        # 평가모드
        model.eval()
        
        # 문장 토크나이징
        tokenized_sent = tokenizer(
            sent,
            return_tensors = 'pt',
            truncation=True,
            add_special_tokens =True,
            max_length = 256
        )
        
        #gpu
        tokenized_sent.to(device)
        
        # 예측
        
        with torch.no_grad():
            outputs = model(
                input_ids = tokenized_sent['input_ids'],
                attention_mask = tokenized_sent['attention_mask'],
                token_type_ids = tokenized_sent['token_type_ids']
            )

        #  결과
        logits = outputs[0]
        logits = logits.detach().cpu()
        result = logits.argmax(-1)

        if result == 0:
            print('부정 리뷰')
            result = 0
        elif result == 1:
            print('긍정 리뷰')
            result = 1
        return result



sent = '씨발'
sentence_predict(sent)