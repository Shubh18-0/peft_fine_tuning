import transformers
from transformers import AutoTokenizer
from config.default_config import MODEL_NAME

class Tokenizer:
    def __init__(self):
        self.model_name=MODEL_NAME
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name)
    
    def tokenize_data(self,dataset_dict):
        def tokenizer_fn(batch):
            data=[str(x) for x in batch['Open-ended Verifiable Question']]
            return self.tokenizer(data,
                             padding='max_length',
                             truncation=True,
                             max_length=128)
        
        mapped_ds=dataset_dict.map(tokenizer_fn,batched=True)
        return mapped_ds

