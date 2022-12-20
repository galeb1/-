import re
from soynlp.normalizer import repeat_normalize

from ElectraDataModule import *
from ElectraBinaryClassification import *

model = ElectraClassification.load_from_checkpoint('epoch=02-val_accuracy=0.876.ckpt')
tokenizer = ElectraTokenizer.from_pretrained("monologg/electra-small-finetuned-imdb")
    
def infer(x, model=model, tokenizer =tokenizer) :

    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fa-zA-Z]+')
    processed = pattern.sub(' ', x)
    processed = processed.strip()
    processed = repeat_normalize(processed, num_repeats=2)

    tokenized = tokenizer(processed, return_tensors='pt')

    output = model(tokenized.input_ids, tokenized.attention_mask)
    
    logits = output[0]
    logits = logits.detach().cpu()
    result = logits.argmax(-1)
    print(result)
    
    if result == 0:
        print('부정 리뷰')
        result = 0
        
    elif result == 1:
        print('긍정 리뷰')
        result = 1
        
    return result

text = 'this is good'
print(infer(text))