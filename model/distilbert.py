from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
from model.common_utils import Node2Vec
import torch.nn as nn
import torch


class DistilBertNode2Vec(Node2Vec):
    
    def __init__(self, abstract=None, pre_tokenize='./data/pre_tokenize_distilbert.pth', device='cuda:0'):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel(DistilBertConfig(n_layers=3, n_heads=4)).to(device)
        if abstract is not None:
            self.abs_ids = self.tokenizer(abstract, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            torch.save(self.abs_ids, pre_tokenize)
        else:
            self.abs_ids = torch.load(pre_tokenize).to(device)
            print("Load pre-tokenized data from", pre_tokenize)
        
    def get_ids_by_idx(self, idx):
        idx_ids = {k:v[idx] for k, v in self.abs_ids.items()}
        return idx_ids
    
    def tokenize_for_inference(self, abstract):
        inputs = self.tokenizer(abstract, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.bert.device)
        return inputs

    def forward(self, node_id):
        inputs = self.get_ids_by_idx(node_id)
        outputs = self.bert(**inputs)

        return outputs.last_hidden_state[:, 0]  # Use the first token (CLS token) as the pooled output



if __name__ == "__main__":
    from dataset.dataloader import _load_titleabs
    titleabs = _load_titleabs()
    
    abst = titleabs['abs'][:].to_list()
    model = DistilBertNode2Vec(abst, device='cuda:0')
    output = model([0, 1, 2, 3, 4])
    print(output.shape)
