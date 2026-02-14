import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr
from pathlib import Path

TOKENIZER_DIR = Path("tokenizer") 
BASE_MODEL_DIR = Path("TinyLlama/TinyLlama-1.1B-Chat-v1.0") 
PEFT_MODEL_DIR = Path("model")  

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(
    base_model,
    PEFT_MODEL_DIR,
    device_map="auto"
)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def answer_question(question):
    inputs = tokenizer(question, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.Interface(fn=answer_question, inputs="text", outputs="text").launch()
