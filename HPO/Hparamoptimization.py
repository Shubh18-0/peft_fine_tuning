import optuna
from config.default_config import default_params
from config.default_config import MODEL_NAME
from  model_methods.model_load import load_model
from utils.compute_metrics_fn import compute_metrics
from utils.datacollator import datacollator
from utils.datasets import Load_dataset
from utils.tokenizer import Tokenizer
import torch
from torch import nn
import transformers
from transformers import Trainer,TrainingArguments,EvalPrediction
from model_methods.model_load import Model_load
import json

class Hparamoptimization:
    def __init__(self):
        self.base_model=Model_load()
        self.tokenizer_obj=Tokenizer()
        self.tokenizer=self.tokenizer_obj.tokenizer
        self.final_model=None
        self.args=None
        
    def load_model(self):
        self.final_model=self.base_model.quantized_model()
        return self.final_model

    def objective(self,trial):
        lora_params={
            'r':trial.suggest_int('lora_rank',4,8),
            'lora_dropout':trial.suggest_float('lora_dropout',0.0,0.1),
            'learning_rate':trial.suggest_float('learning_rate',1e-5,3e-4,log=True),
            'weight_decay':trial.suggest_float('weight_decay',0.0,0.1),
            'lora_alpha':trial.suggest_categorical('lora_alpha',[8,16])
        }
        return lora_params

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
        self.args=TrainingArguments(
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
            num_train_epochs=default_params['num_train_epochs']

        )
        return self.args
    
    def HPO_trainer(self):
        if self.final_model is None:
            self.load_model()
        if self.args is None:
            self.training_args()
        tokenized_ds = self.tokenize_ds()
        trainer=Trainer(
            args=self.args,
            model=self.final_model,
            train_dataset=tokenized_ds['train'],
            eval_dataset=tokenized_ds['validation'],
            compute_metrics=self.computed_metrics,
            data_collator=self.collated_ds()
        )
        trainer.train()
        eval_result=trainer.evaluate()
        return eval_result['eval_loss']


hyperparameter_search=Hparamoptimization()
study=optuna.create_study(direction='minimize')
study.optimize(hyperparameter_search.objective,n_trials=3)
print('Best hyperparameters':study.best_params)
best_params=study.best_params

with open('best_params.json','w') as f:
    json.dump(study.best_params,f)
    