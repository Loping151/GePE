from transformers import AutoTokenizer, AutoModel
from model.common_utils import Node2Vec
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, nhead, nhid, nlayers, dropout=0.5):
        super(SimpleTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = TransformerEncoderLayer(input_dim, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(10000, input_dim)  # Assuming a vocab size of 10000
        self.input_dim = input_dim
        self.decoder = nn.Linear(input_dim, 2)  # Assuming binary classification

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.input_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:,0,:])
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
# NOTE: Not finished yet
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