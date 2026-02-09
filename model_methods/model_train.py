import torch 
from torch import nn
from torch import trainer,TrainingArguments

class Trainer_model:
    def __init__(self):
        self.TrainingArguments=None

    def training_args(self):
        args=TrainingArguments(

        )

        return args
    
    def trainer(self):
        final_model=trainer(args=self.training_args(),
                            )