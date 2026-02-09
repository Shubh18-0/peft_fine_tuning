import peft
from peft import get_peft_model , LoraConfig
from transformers import AutoModelForCausalLM
from config.default_config import MODEL_NAME
from config.lora_config import lora_default_configs
from transformers import BitsAndBytesConfig
import torch
from torch import nn

class Model_load:
    def __init__(self):
        self.model_name=MODEL_NAME

    def load_quantization(self):
        bnb_config=BitsAndBytesConfig(
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        return bnb_config

    def load_model(self):
        model=AutoModelForCausalLM.from_pretrained(self.model_name,
                                   quantization_config=self.load_quantization(),
                                   device_map='auto')
        return model
    
    def lora_configs(self):
        lora_config=LoraConfig(
            task_type=lora_default_configs['task_type'],
            lora_alpha=lora_default_configs['lora_alpha'],
            r=lora_default_configs['lora_rank'],
            lora_dropout=lora_default_configs['lora_dropout'],
            bias=None
        )
        return lora_config
    
    def quantized_model(self):
        q_model=get_peft_model(self.load_model(),
                               self.lora_configs())
        return q_model