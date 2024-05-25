from dataset.dataloader import arxiv_dataset, load_titleabs
from dataset.embedding import get_scibert, encode_single
from model.common_utils import Classifier
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.common_utils import evaluate
from model.bert import BertNode2Vec
from model.scibert import SciBertNode2Vec
from model.embedding import Embedding
from model.hashmlp import MLP
from tqdm import tqdm
from utils.args import get_app_args 
import torch.nn.functional as F
from app.get_abstract import title_to_abs

from sklearn.neighbors import NearestNeighbors



args = get_app_args()

device = args.device
data = arxiv_dataset()
titleabs = load_titleabs()

with torch.no_grad():
    # emb = torch.load('./data/embeddings_cls.pth').cpu()

    if args.model_type == 'bert':
        model = BertNode2Vec(device=device)
        model.load(args.pretrain, device)
        emb = model.embed_all(data)
    elif args.model_type == 'scibert':
        model = SciBertNode2Vec(device=device)
        model.load(args.pretrain, device)
        emb = model.embed_all(data)
        
F.normalize(emb, p=2, dim=1)

knn = NearestNeighbors(n_neighbors=args.k, metric='cosine')
knn.fit(emb.numpy())

while True:
    try:
        abstract = title_to_abs()
    except Exception as e:
        print(e, '\nTry again.')
        continue

    if not abstract:
        continue
    else:
        inf_emb = model.inference(abstract).cpu()
        # tokenizer, model = get_scibert()
        # inf_emb = encode_single(abstract, tokenizer, model).cpu()
        inf_emb = F.normalize(inf_emb, p=2, dim=1)
        
        distances, nearest_indices = knn.kneighbors(inf_emb.numpy(), return_distance=True)

        print(f"The indices of the {args.k} nearest nodes are: {nearest_indices[0]}")
        print(f"The distances are: {distances[0]}")
        
        print('\n\nRecommendation:\n')
        print('-----------------')
        for idx in nearest_indices[0]:
            print(titleabs['title'][idx])
            print(titleabs['abs'][idx])
            print('-----------------')
        print('\n\n')