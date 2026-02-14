from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torch import nn
import peft
from transformers import AutoModelForCausalLM , AutoTokenizer
from peft import PeftModel
app=FastAPI()

model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0'
tokenizer=AutoTokenizer.from_pretrained(model_name)

base_model=AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    load_in_4bit=True,
    torch_dtype=torch.float16
)

saved_adapter_path=r'C:\Users\bhard\Desktop\tunedmodelfiles\model'
print('Loading Base model... ')
print('Loading adapters for QLORA...')
model=PeftModel.from_pretrained(base_model,
                     adapter_name=saved_adapter_path)

model.eval()

class Prompt(BaseModel):
    text:str

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
@app.post('/predict')
def generate_answers(data:Prompt):
    inputs=tokenizer(data.text,return_tensors='pt').to(device)
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=300)
    return {'response':tokenizer.decode(output[0],skip_special_tokens=True)}

@app.get('/')
def health_check():
    return {'message':
            'App running fine :)'}
   
