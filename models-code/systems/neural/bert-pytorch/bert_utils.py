import torch
from torch.utils.data import Dataset

class BertDataset(Dataset):
    def __init__(self, texts, labels,
                tokenizer, max_length):
        super(BertDataset, self).__init__()
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.texts=texts
        self.labels=labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        text1 = self.texts[index]
        inputs = self.tokenizer.encode_plus(
            text = text1 ,
            text_pair = None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        output =  {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'target': torch.tensor(self.labels[index], dtype=torch.long),
            }

        if "token_type_ids" in inputs:
            token_type_ids = inputs["token_type_ids"]
            output['token_type_ids'] = torch.tensor(token_type_ids, dtype=torch.long)
        return output
