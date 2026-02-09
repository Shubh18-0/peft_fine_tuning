from transformers import DataCollatorForLanguageModeling

def datacollator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)