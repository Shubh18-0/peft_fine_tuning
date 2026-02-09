import wandb
import os
from dotenv import load_dotenv
load_dotenv()

def wandb_login():
    api_key=os.getenv('wandb_api_key')

    if api_key:
        wandb.login(key=api_key)

    else:
        raise ValueError("Wand b Api key not found . Please setup an api key to log metrics")
    
def wandb_init(project_name='final_tuning',
               name=None):
    wandb.init(project=project_name)