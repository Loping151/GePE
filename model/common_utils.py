from ogb.nodeproppred import Evaluator
from transformers import BertConfig, BertTokenizer, BertModel
import torch.nn as nn
import torch


class Classifier(nn.Module):
    def __init__(self, in_dim, num_cls):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_cls)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc(x)
        # x = self.softmax(x)
        return x


def evaluate(pred, label):
    input_dict = {"y_true": label, "y_pred": pred}
    return Evaluator(name='ogbn-arxiv').eval(input_dict)


class NegativeSamplingLoss(nn.Module):
    """The negative sampling loss function.

    Args:
        eps (float, optional): For numerical stability. Defaults to 1e-7.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        cur_embs: torch.Tensor,
        pos_embs: torch.Tensor,
        neg_embs: torch.Tensor,
    ):
        """
        Compute the negative sampling loss.

        Args:
            cur_embs (torch.Tensor): Embeddings of the current nodes, shape (B, H).
            pos_embs (torch.Tensor): Embeddings of the positive samples, shape (B, N_pos, H).
            neg_embs (torch.Tensor): Embeddings of the negative samples, shape (B, N_neg, H).

        Returns:
            torch.Tensor: The negative sampling loss.
        """
        # Reshape embeddings for broadcasting
        B, H = cur_embs.shape
        cur_embs = cur_embs.unsqueeze(1)  # shape (B, 1, H)
        pos_embs = pos_embs.view(B, -1, H)  # shape (B, N_pos, H)
        neg_embs = neg_embs.view(B, -1, H)  # shape (B, N_neg, H)

        # Compute scores
        pos_scores = torch.bmm(pos_embs, cur_embs.transpose(1, 2)).squeeze(2)  # shape (B, N_pos)
        neg_scores = torch.bmm(neg_embs, cur_embs.transpose(1, 2)).squeeze(2)  # shape (B, N_neg)

        # Compute positive and negative losses
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + self.eps).sum()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + self.eps).sum()

        # Total loss
        loss = pos_loss + neg_loss

        return loss
