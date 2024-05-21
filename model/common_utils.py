from ogb.nodeproppred import Evaluator
import torch.nn as nn
import torch
import torch.nn.functional as F



class Node2Vec(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path):
        """Save the model parameters to the specified path."""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path, *args, **kwargs):
        """Load the model parameters from the specified path."""
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        print(f"Model loaded from {path}")
        return model


class Classifier(nn.Module):
    def __init__(self, in_dim, num_cls):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_cls)
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc(x)
        # x = self.sigmoid(x)
        return x


def evaluate(pred, label):
    input_dict = {"y_true": label, "y_pred": pred}
    return Evaluator(name='ogbn-arxiv').eval(input_dict)


class NegativeSamplingLoss(nn.Module):
    """The negative sampling loss function.

    Args:
        eps (float, optional): For numerical stability. Defaults to 1e-9.
    """

    def __init__(self, eps: float = 1e-9):
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
        cur_embs = F.normalize(cur_embs, p=2, dim=1)  # shape (B, H)
        pos_embs = F.normalize(pos_embs, p=2, dim=2)  # shape (B, N_pos, H)
        neg_embs = F.normalize(neg_embs, p=2, dim=2)  # shape (B, N_neg, H)

        # Reshape embeddings for broadcasting
        cur_embs = cur_embs.unsqueeze(1)  # shape (B, 1, H)
        
        # Compute scores. Actually cos_sim
        pos_scores = torch.bmm(pos_embs, cur_embs.transpose(1, 2)).squeeze(2)  # shape (B, N_pos)
        neg_scores = torch.bmm(neg_embs, cur_embs.transpose(1, 2)).squeeze(2)  # shape (B, N_neg)

        # Compute positive and negative losses
        pos_loss = (1-pos_scores).mean()
        neg_loss = neg_scores.mean()

        # Total loss
        loss = pos_loss + neg_loss

        return loss

# Example usage
if __name__ == "__main__":
    loss_fn = NegativeSamplingLoss()
    cur_embs = torch.randn(8, 128)  # Example current node embeddings
    pos_embs = torch.randn(8, 5, 128)  # Example positive samples
    neg_embs = torch.randn(8, 20, 128)  # Example negative samples

    loss = loss_fn(cur_embs, pos_embs, neg_embs)
    print("Loss:", loss.item())