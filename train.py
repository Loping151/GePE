import torch
import torch.optim as optim
import warnings

from tqdm import tqdm
from torch.utils.data import DataLoader

from model.bertnode2vec import BertNode2Vec, NegativeSamplingLoss
from utils.walker import BiasedRandomWalker


class BertNode2VecTrainer:
    """A trainer class for training the `BertNode2Vec` model.

    Args:
    - `num_nodes` (int): Total number of nodes in the graph.
    - `model` (Node2Vec): A `Node2Vec` model instance to be trained.
    - `walker` (BiasedRandomWalker): A random walker.
      This walker should implement:
        1. a `walk(start, length)` method that returns a walk of length `length` starting from `start`.
        2. a `connected_nodes` attribute that lists all nodes with at least one edge.
    - `n_negs` (int): Number of negative samples to be used in negative sampling.
    - `n_epochs` (int): Number of epochs to train the model.
    - `batch_size` (int): Batch size for training.
    - `lr` (float): Learning rate for training.
    - `device` (torch.device): Device to run the training.
    - `walk_length` (int): Length of each random walk session. Defaults to 15.
    - `window_size` (int): Window size for each training sample Defaults to 7.
    - `n_walks_per_node` (int): Number of walks to start from each node. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes: int,
        model: BertNode2Vec,
        walker: BiasedRandomWalker,
        n_negs: int,
        n_epochs: int,
        batch_size: int,
        lr: float,
        device: torch.device,
        walk_length: int = 15,
        window_size: int = 7,
        n_walks_per_node: int = 1,
    ):
        self.num_nodes = num_nodes
        self.model = model
        self.n_negs = n_negs

        self.walker = walker
        self.walk_length = walk_length
        self.n_walks_per_node = n_walks_per_node

        if window_size % 2 == 0:
            warnings.warn("Window size should be odd. Adding 1 to window size.")
        self.window_size = (window_size // 2) * 2 + 1

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.optimizer = self.create_optimizer(lr)
        self.loss_func = NegativeSamplingLoss()

    def _get_random_walks(self):
        """
        Performs random walks using the `walker`,
        converts the walks into training samples,
        and returns a wrapped `DataLoader` for training.
        """
        walk_len = self.walk_length
        context_sz = self.window_size // 2

        # Perform random walks starting from each node in `connected_nodes`
        trajectories = []
        for node in tqdm(self.walker.connected_nodes):
            for _ in range(self.n_walks_per_node):
                trajectory = self.walker.walk(node, walk_len)
                trajectories.append(trajectory)

        # Convert the walks into training samples
        walks = []
        for trajectory in trajectories:
            for center in range(len(trajectory)):
                start = max(0, center - context_sz)
                end = min(len(trajectory), center + context_sz + 1)
                context = trajectory[start:center] + trajectory[center+1:end]
                walks.append((trajectory[center], context))

        return DataLoader(walks, batch_size=self.batch_size, shuffle=True)

    def _sample_neg_nodes(self, batch_sz: int, context_sz: int, n_negs: int):
        """Returns a batch of negative samples, to be used for NegativeSamplingLoss.

        Args:
        - batch_sz (int): Batch size.
        - context_sz (int): Context size.
        - n_negs (int): Number of negative samples to be used.

        NOTE: We simply randomly sample from all nodes and ignore the fact that
        we might accidentally include positive edges during sampling.
        Since the graph is sparse, this should not cause much trouble.
        """
        return torch.randint(self.num_nodes, (batch_sz, context_sz * n_negs))

    def _train_one_epoch(self, eid: int):
        """
        Perform one epoch of training.
        We first perform random walks to generate training samples,
        then train the model using these samples.
        """
        tot_loss = 0
        prog = tqdm(self._get_random_walks())
        for bid, batch in enumerate(prog):
            self.optimizer.zero_grad()

            batch = [(center, context) for center, context in batch]
            currents = torch.LongTensor([center for center, _ in batch]).to(self.device)
            contexts = torch.LongTensor([context for _, context in batch]).to(self.device)

            B = currents.shape[0]  # batch size
            L = contexts.shape[1]  # context size

            # Current node embeddings
            cur_embeddings = self.model(currents)  # (B, D)

            # Positive samples
            pos_embeddings = self.model(contexts.view(-1))  # (B * L, D)
            pos_embeddings = pos_embeddings.view(B, L, -1)  # (B, L, D)

            # Negative samples, choose L * n_negs neg samples for each node
            neg_nodes = self._sample_neg_nodes(B, L, self.n_negs).to(self.device)  # (B, L * n_negs)
            neg_embeddings = self.model(neg_nodes.view(-1))  # (B * L * n_negs, D)
            neg_embeddings = neg_embeddings.view(B, L * self.n_negs, -1)  # (B, L * n_negs, D)

            loss = self.loss_func(cur_embeddings, pos_embeddings, neg_embeddings)

            loss.backward()
            self.optimizer.step()

            tot_loss += loss.item()
            avg_loss = tot_loss / (bid + 1)

            prog.set_description(f"Epoch: {eid:2d}, Loss: {avg_loss:.4f}")

        print(f"Epoch: {eid:2d}, Loss: {avg_loss:.4f}")

    def create_optimizer(self, lr: float):
        """Create an optimizer for training."""
        return optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        """Train the model for `n_epochs` epochs."""

        self.model.train()
        for eid in range(self.n_epochs):
            self._train_one_epoch(eid)


if __name__ == "__main__":
    from dataset.dataloader import arxiv_dataset
    data = arxiv_dataset()
    walker = BiasedRandomWalker(data['graph'])

    model = BertNode2Vec(device='cuda')

    trainer = BertNode2VecTrainer(
        num_nodes=data['graph'].num_nodes,
        model=model,
        walker=walker,
        n_negs=5,
        n_epochs=10,
        batch_size=256,
        lr=0.01,
        device='cuda',
        walk_length=15,
        window_size=7,
        n_walks_per_node=10
    )

    trainer.train()
