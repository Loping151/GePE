import random
from typing import List, Tuple
from torch_geometric.utils import degree
import torch
import torch.sparse

class BiasedRandomWalker:
    """
    A biased random walker for generating random walks on a graph.
    """

    def __init__(self, data, p: float = 1.2, q: float = 2.0, device='cuda:0'):
        self.data = data
        self.ret_p = p
        self.io_q = q
        self.device = device

        self.connected_nodes = self._get_connected_nodes()
        self.edge_index = self.data.edge_index
        self.num_nodes = self.data.num_nodes
        self.sparse_adj = self._convert_to_sparse_matrix()

    def _convert_to_sparse_matrix(self):
        """Convert edge_index to a sparse adjacency matrix."""
        row, col = self.edge_index
        adj = torch.sparse_coo_tensor(torch.stack([row, col]), torch.ones_like(row), (self.num_nodes, self.num_nodes))
        return adj.to(self.device)

    def _get_connected_nodes(self):
        """
        Returns a list of nodes that have at least one edge connected to them.
        """
        deg = degree(self.data.edge_index[0], self.data.num_nodes)
        connected_nodes = deg.nonzero(as_tuple=False).view(-1).tolist()
        print(f"Number of connected nodes: {len(connected_nodes)}")
        return connected_nodes

    def _normalize(self, weights):
        """Normalizes the weights to make them sum to 1."""
        tot = sum(weights)
        return [p / tot for p in weights]

    def get_probs_uniform(self, curr_node) -> Tuple[List[int], List[float]]:
        """Returns a normalized uniform probability distribution
        over the neighbors of the current node
        """
        indices = self.sparse_adj._indices()
        # values = self.sparse_adj._values()
        nexts = indices[1][indices[0] == curr_node].tolist()
        probs = [1 / len(nexts)] * len(nexts)
        return nexts, probs

    def get_probs_biased(self, curr_node, prev_node: int) -> Tuple[List[int], List[float]]:
        """Returns a normalized biased probability distribution
        over the neighbors of the current node
        """
        indices = self.sparse_adj._indices()
        # values = self.sparse_adj._values()
        curr_nbrs = indices[1][indices[0] == curr_node]

        nexts = []
        unnormalized_probs = []
        for next in curr_nbrs:
            nexts.append(next)

            if next == prev_node:
                unnormalized_probs.append(1 / self.ret_p)
            elif next in indices[1][indices[0] == prev_node]:
                unnormalized_probs.append(1)
            else:
                unnormalized_probs.append(1 / self.io_q)

        # normalize the probabilities
        probs = self._normalize(unnormalized_probs)
        return nexts, probs

    def walk(self, start: int, length: int) -> List[int]:
        """Perform a random walk of length `length`, starting from node `start`.

        Args:
            start (int): The node id to start the random walk.
            length (int): The length of the random walk.

        Returns:
            List[int]: A list of node ids representing the random walk trajectory.
        """

        trace = [start]
        current_len = 1
        prev = None

        while current_len < length:
            curr_node = trace[-1]

            if prev is None:
                # For the first node, sample uniformly at random
                nexts, probs = self.get_probs_uniform(curr_node)
            else:
                # For the subsequent nodes, sample based on the biased probabilities
                nexts, probs = self.get_probs_biased(curr_node, prev)

            if not nexts:
                break  # If no neighbors, end the walk

            # `target` is to be sampled from neighboring nodes based on probabilities
            target = random.choices(nexts, probs)[0]
            trace.append(target)

            prev = curr_node
            current_len += 1

        return trace


if __name__ == "__main__":
    from dataloader import arxiv_dataset
    
    data = arxiv_dataset()
    walker = BiasedRandomWalker(data['graph'])

    start_node = data['train_idx'][222].item()
    walk_length = 10
    random_walk = walker.walk(start_node, walk_length)
    print("Random walk:", random_walk)
