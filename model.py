from ogb.nodeproppred import Evaluator
from transformers import BertConfig, BertTokenizer, BertModel
import torch.nn as nn
import torch


class bert_node2vec(nn.Module):
    
    def __init__(self, abstract=None, pre_tokenize='./data/pre_tokenize.pth', device='cuda:0'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel(BertConfig()).to(device)
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

        return outputs.pooler_output


class classifier(nn.Module):
    def __init__(self, in_dim, num_cls):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_cls)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


def evaluate(pred, label):
    input_dict = {"y_true": label, "y_pred": pred}
    return Evaluator(name='ogbn-arxiv').eval(input_dict)

if __name__ == "__main__":
    from dataloader import _load_titleabs
    titleabs = _load_titleabs()
    
    abs_10 = titleabs['abs'][:].to_list()
    model = bert_node2vec(abs_10, device='cuda:0')
    output = model([0, 1, 2, 3, 4])
    print(output.shape)
