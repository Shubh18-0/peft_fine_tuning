from transformers import EvalPrediction
import torch
from torch import nn

def compute_metrics(eval_pred:EvalPrediction):
    logits,labels=eval_pred
    shift_logits=torch.tensor(logits[...,:-1,:]).reshape(-1,logits.shape[-1])
    shift_labels=torch.tensor(labels[...,1:]).reshape(-1)
    loss=nn.CrossEntropyLoss()
    loss_shifted=loss(shift_logits,shift_labels).item()

    return {
        'loss':loss_shifted
    }
    
