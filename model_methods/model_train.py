import torch 
from torch import nn
from transformers import Trainer,TrainingArguments
from config.default_config import default_params
import optuna
from config.default_config import MODEL_NAME
from  model_methods.model_load import load_model
from utils.compute_metrics_fn import compute_metrics
from utils.datacollator import datacollator
from utils.datasets import Load_dataset
from utils.tokenizer import Tokenizer
import peft
from model_methods.model_load import Model_load
from peft import get_peft_model,LoraConfig
import json
from transformers import BitsAndBytesConfig

def load_best_params():
    with open('best_params.json','r') as f:
        best_params=json.load(f)
    return best_params

class Trainer_model:
    def __init__(self,best_params=load_best_params()):
        self.trainingargs=None
        self.base_model=Model_load()
        self.tokenizer_obj=Tokenizer()
        self.tokenizer=self.tokenizer_obj.tokenizer
        self.final_model=None
        self.args=None
        self.best_params=best_params
        
    def load_model(self):
        self.final_model=self.base_model.quantized_model()
        return self.final_model
    
    def loading_dataset(self):
        ds_loader=Load_dataset()
        ds=ds_loader.load_dataset()
        split_ds=ds_loader.split_dataset(dataset=ds)
        train_ds,validation_ds,test_ds=split_ds
        ds_dict=ds_loader.dataset_dict(train_ds=train_ds,
                                          validation_ds=validation_ds,
                                          test_ds=test_ds)
        return ds_dict
    
    def tokenize_ds(self):
        ds_dict=self.loading_dataset()
        tokenized_dataset=self.tokenizer_obj.tokenize_data(ds_dict)

        return tokenized_dataset
    
    def collated_ds(self):
        self.collated_dataset=datacollator()
        return self.collated_dataset
    
    def computed_metrics(self,eval_pred:EvalPrediction):
        self.computed_results=compute_metrics(eval_pred)
        return self.computed_results

    def training_args(self):
            self.trainingargs=TrainingArguments(
            output_dir=default_params['output_dir'],
            logging_dir=default_params['logging_dir'],
            per_device_train_batch_size=default_params['per_device_train_batch_size'],
            per_device_eval_batch_size=default_params['per_device_eval_batch_size'],
            save_strategy=default_params['save_strategy'],
            eval_strategy=default_params['eval_strategy'],
            fp16=True,
            gradient_accumulation_steps=default_params['gradient_accumulation_steps'],
            gradient_checkpointing=True,
            save_steps=default_params['save_steps'],
            eval_steps=default_params['eval_steps'],
            report_to=default_params['report_to'],
            load_best_model_at_end=True,
            save_total_limit=default_params['save_total_limit'],
            num_train_epochs=default_params['num_train_epochs'],
            learning_rate=self.best_params['learning_rate'],
            weight_decay=self.best_params['weight_decay']
        ) 
            return self.trainingargs
    
    def create_final_model(self):
        if self.final_model is None:
            self.load_model()

        lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=self.best_params['lora_rank'],
        lora_alpha=self.best_params['lora_alpha'],
        lora_dropout=self.best_params['lora_dropout'],
        bias="none")
        
        self.final_model=get_peft_model(self.final_model,
                                   lora_config
                                   )
        return self.final_model
    def final_model_trainer(self):
        tokenized_ds = self.tokenize_ds()
        trainer=Trainer(
            args=self.trainingargs,
            model=self.final_model,
            train_dataset=tokenized_ds['train'],
            eval_dataset=tokenized_ds['validation'],
            compute_metrics=self.computed_metrics,
            data_collator=self.collated_ds
        )
        trainer.train()
        eval_result=trainer.evaluate()
        return eval_result['eval_loss']

         