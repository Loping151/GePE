import torch
import torch.optim as optim
import warnings

import random
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.distilbert import DistilBertNode2Vec
from model.embedding import Embedding
from model.hashmlp import MLP
from model.common_utils import NegativeSamplingLoss
from utils.walker import BiasedRandomWalker
from utils.walker_parallel import parallel_run, get_walks_single
import os, shutil
import math
from time import strftime

            
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
        model,
        walker: BiasedRandomWalker,
        n_negs: int,
        n_epochs: int,
        batch_size: int,
        lr: float,
        device: torch.device,
        num_workers: int = 12,
        walk_length: int = 6,
        window_size: int = 5,
        n_walks_per_node: int = 3,
        sample_node_prob: float = 0.02, 
        save_path: str = f'./checkpoint/{strftime("%Y%m%d%H%M%S")}'
    ):
        self.model = model
        self.n_negs = n_negs

        self.walker = walker
        self.walk_length = walk_length
        self.n_walks_per_node = n_walks_per_node
        self.sample_node_prob = sample_node_prob

        if window_size % 2 == 0:
            warnings.warn("Window size should be odd. Adding 1 to window size.")
        self.window_size = (window_size // 2) * 2 + 1

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.optimizer = self.create_optimizer(lr)
        self.loss_func = NegativeSamplingLoss()
        
        self.num_nodes = math.floor(self.walker.num_nodes * self.sample_node_prob)
        self.num_workers = num_workers
        self.save_path = save_path
        
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
        for node in tqdm(random.sample(self.walker.connected_nodes, self.num_nodes)):
            for _ in range(self.n_walks_per_node):
                trajectory = self.walker.walk(node, walk_len)
                trajectories.append(trajectory)

        # Convert the walks into training samples
        walks = []
        for trajectory in trajectories:
            for cent in range(context_sz, walk_len - context_sz):
                walks.append(trajectory[cent - context_sz : cent + context_sz + 1])
        walks = torch.tensor(walks)
        
        return DataLoader(walks, batch_size=self.batch_size, shuffle=True)


    def _get_random_walks_parallel(self):
        total_nodes = self.walker.num_nodes
        batch_size = total_nodes // self.num_workers + 1
        params = [(start, min(start + batch_size, total_nodes)) for start in range(0, total_nodes, batch_size)]
        args = [self.walk_length, self.window_size // 2, self.n_walks_per_node, self.walker]
        if os.path.exists('.tmp'):
            shutil.rmtree('.tmp')
        os.mkdir('.tmp')
        
        parallel_run(get_walks_single, params, args, num_workers=self.num_workers)
        walks = []
        for start, end in params:
            walks.append(torch.load(f'.tmp/walks_{start}_{end}.pth'))
        
        walks = torch.cat(walks, dim=0).type(torch.long)
        shutil.rmtree('.tmp')
        
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
        return torch.randint(self.walker.num_nodes, (batch_sz, context_sz * n_negs))

    def _train_one_epoch(self, eid: int):
        """
        Perform one epoch of training.
        We first perform random walks to generate training samples,
        then train the model using these samples.
        """
        tot_loss = 0
        print("Walking...")
        prog = tqdm(self._get_random_walks()) if self.num_nodes < 1000*self.num_workers else tqdm(self._get_random_walks_parallel())
        context_sz = self.window_size // 2
        for bid, batch in enumerate(prog):
            self.optimizer.zero_grad()

            batch = batch.to(self.device)

            B = batch.shape[0]  # batch size
            L = batch.shape[1] - 1  # context size
            
            currents = batch[:, context_sz]
            contexts = torch.cat((batch[:, :context_sz], batch[:, context_sz+1:]), dim=1).contiguous()

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

            prog.set_description(f"Epoch: {eid:2d}, avg_loss: {avg_loss:.4f}, loss: {loss.item():.4f}")
        self.model.save(f'{self.save_path}/model_{eid}.pth')
        with open(f'{self.save_path}/loss.txt', 'a') as f:
            f.write(f"{avg_loss}\n")

        print(f"Epoch: {eid:2d}, Loss: {avg_loss:.4f}")

    def create_optimizer(self, lr: float):
        """Create an optimizer for training."""
        return optim.AdamW(self.model.parameters(), lr=lr)

    def train(self):
        """Train the model for `n_epochs` epochs."""

        self.model.train()
        for eid in range(self.n_epochs):
            self._train_one_epoch(eid)


if __name__ == "__main__":
    from utils.args import get_train_args
    from dataset.dataloader import arxiv_dataset
    
    data = arxiv_dataset()
    walker = BiasedRandomWalker(data['graph'])

    args = get_train_args()
    
    if args.model_type == 'bert':
        model = DistilBertNode2Vec(device=args.device)
    elif args.model_type == 'embedding':
        model = Embedding(data['graph'].num_nodes, 768, device=args.device)
    elif args.model_type == 'mlp':
        model = MLP(in_dim=132, embedding_dim=768, hidden=16384, device=args.device)

    if args.pretrain is not None:
        model = model.load(args.pretrain, args.device)

    trainer = BertNode2VecTrainer(
        model=model,
        walker=walker,
        n_negs=args.n_negs,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        walk_length=args.walk_length,
        window_size=args.window_size,
        n_walks_per_node=args.n_walks_per_node,
        sample_node_prob=args.sample_node_prob,
        save_path=args.save_path
    )

    trainer.train()