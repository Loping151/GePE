from dataset.dataloader import arxiv_dataset, ClassifierDataset
from model.common_utils import Classifier
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.common_utils import evaluate
from model.distilbert import DistilBertNode2Vec
from dataset.embedding import encode_single
from tqdm import tqdm
from utils.args import get_vaildate_args


if __name__ == "__main__":
    args = get_vaildate_args()
    
    device = args.device
    
    data = arxiv_dataset()
    label_train = data['graph'].y[np.concatenate([data['train_idx'], data['valid_idx']])]
    label_test = data['graph'].y[data['test_idx']]
    
    with torch.no_grad():
        if args.model_type == 'scibert':
            emb = torch.load('./data/embeddings_cls.pth')
        
        elif args.model_type == 'randombert':
            model = DistilBertNode2Vec(device=device)

            emb_list = []
            
            for ids in tqdm(range(0, data['graph'].num_nodes)):
                emb = model([ids]).cpu()
                emb_list.append(emb)
            emb = torch.cat(emb_list, dim=0)
            
        elif args.model_type == 'distilbert':
            model = DistilBertNode2Vec(device=device)
            model.load_state_dict(torch.load(args.pretrain))

            emb_list = []
            
            for ids in tqdm(range(0, data['graph'].num_nodes)):
                emb = model([ids]).cpu()
                emb_list.append(emb)
            emb = torch.cat(emb_list, dim=0)
    
    emb_train = emb[np.concatenate([data['train_idx'], data['valid_idx']])]
    emb_test = emb[data['test_idx']]
    
    classifier = Classifier(in_dim=emb.shape[1], num_cls=40).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=args.lr)
    
    train_loader = DataLoader(ClassifierDataset(emb_train, label_train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(ClassifierDataset(emb_test, label_test), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    
    for epoch in range(args.num_epochs):
        classifier.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()

            optimizer.zero_grad()
            outputs = classifier(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], avg_loss: {avg_loss:.4f}, loss: {loss.item():.4f}")

    print("Training completed.")
    
    print("Evaluating...")
    classifier.eval()
    with torch.no_grad():
        pred = []
        true = []
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()
            outputs = classifier(batch_x)
            pred.append(outputs.argmax(dim=1))
            true.append(batch_y)
        pred = torch.cat(pred).cpu().numpy().reshape(-1, 1)
        true = torch.cat(true).cpu().numpy().reshape(-1, 1)
    result = evaluate(pred, true)
    print('Accuracy:', result)