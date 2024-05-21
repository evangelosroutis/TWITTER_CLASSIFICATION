import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TwitterDataset(Dataset):
    def __init__(self,sentences,labels,tokenizer):
 
        self.sentences=sentences 
        self.labels=labels
        self.tokenizer=tokenizer
        

    def __len__(self):
        return len(self.sentences)
        

    def __getitem__(self,index):
        sentence=self.sentences[index]
        label=self.labels[index]
        tokenized_sentence=self.tokenizer.encode(sentence)

        return {
            'input_ids':torch.tensor(tokenized_sentence),
            'label':label
        }

    

def collate_fn(batch):

    input_ids = [item['input_ids'] for item in batch]
    labels = [item['label'] for item in batch]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = (padded_input_ids != 0).long()

    return {
        'input_ids': padded_input_ids,
        'attention_masks': attention_masks,
        'labels': labels
    }



def create_dataloader(sentences, labels, tokenizer, batch_size, shuffle=True, drop_last=True):
    dataset=TwitterDataset(sentences,labels,tokenizer)
    dataloader=DataLoader(dataset,batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn)
    return dataloader
