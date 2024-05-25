from transformers import AutoConfig, AutoTokenizer, AutoModel
from model.common_utils import Node2Vec
import torch.nn as nn
import torch

class TransformerNode2Vec(Node2Vec):
    
    def __init__(self, model_name='bert-base-uncased', abstract=None, pre_tokenize='./data/pre_tokenize_transformer.pth', device='cuda:0'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name).to(device)
        if abstract is not None:
            self.abs_ids = self.tokenizer(abstract, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            torch.save(self.abs_ids, pre_tokenize)
        else:
            self.abs_ids = torch.load(pre_tokenize, map_location=device)
            print("Load pre-tokenized data from", pre_tokenize)
        
    def get_ids_by_idx(self, idx):
        idx_ids = {k:v[idx] for k, v in self.abs_ids.items()}
        return idx_ids
    
    @torch.no_grad()
    def inference(self, abstract):
        inputs = self.tokenizer(abstract, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.transformer.device)
        outputs = self.transformer(**inputs)
        return outputs.last_hidden_state[:, 0]

    def forward(self, node_id):
        inputs = self.get_ids_by_idx(node_id)
        outputs = self.transformer(**inputs)
        return outputs.last_hidden_state[:, 0]  # Use the first token (CLS token) as the pooled output

if __name__ == "__main__":
    from dataset.dataloader import load_titleabs
    titleabs = load_titleabs()
    
    abst = titleabs['abs'][:].to_list()
    model = TransformerNode2Vec(model_name='bert-base-uncased', abstract=abst, device='cuda:0')
    output = model([0, 1, 2, 3, 4])
    print(output.shape)