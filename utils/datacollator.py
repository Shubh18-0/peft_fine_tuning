from transformers import DataCollatorForLanguageModeling

def datacollator(tokenizer):
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return data_collator