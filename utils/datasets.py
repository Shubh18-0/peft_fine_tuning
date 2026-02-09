from datasets import load_dataset,Datasetdict

class Load_dataset:
    def __init__(self,dataset_name='FreedomIntelligence/medical-o1-verifiable-problem'):
        self.dataset_name=dataset_name

    def load_dataset(self):
        ds=load_dataset(self.dataset_name)
        return ds
        
    def split_dataset(self,dataset):
        split1=dataset['train'].train_test_split(test_size=0.2,seed=42)
        train_ds=split1['train']
        split2=split1['test'].train_test_split(test_size=0.5,seed=42)
        validation_ds=split2['train']
        test_ds=split2['test']
        return train_ds,validation_ds,test_ds

    def dataset_dict(self,train_ds,validation_ds,test_ds):
        dataset=Datasetdict({
            'train':train_ds,
            'validation':validation_ds,
            'test':test_ds})
        return dataset
    
