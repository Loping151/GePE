from dataloader import arxiv_dataset, ClassifierDataset
from model import Classifier
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import evaluate

if __name__ == "__main__":
    device = 'cuda:0'
    num_epochs = 1000
    
    
    data = arxiv_dataset()
    label_train = data['graph'].y[np.concatenate([data['train_idx'], data['valid_idx']])].to(device)
    label_test = data['graph'].y[data['test_idx']]
    emb = torch.load('./data/embeddings_cls.pth').to(device)
    emb_train = emb[np.concatenate([data['train_idx'], data['valid_idx']])]
    emb_test = emb[data['test_idx']]
    
    classifier = Classifier(in_dim=emb.shape[1], num_cls=40).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    train_loader = DataLoader(ClassifierDataset(emb_train, label_train), batch_size=2048, shuffle=True)
    test_loader = DataLoader(ClassifierDataset(emb_test, label_test), batch_size=2048, shuffle=False)
    
    
    for epoch in range(num_epochs):
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

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