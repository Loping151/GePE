import argparse
import random
import numpy as np
import torch



def get_train_args():
    parser = argparse.ArgumentParser(description='Train a BertNode2Vec model.')

    # Trainer arguments
    parser.add_argument('--n_negs', type=int, default=3, help='Number of negative samples to be used in negative sampling.')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run the training.')
    parser.add_argument('--num_workers', type=int, default=23, help='Number of workers for parallel processing.')
    parser.add_argument('--walk_length', type=int, default=5, help='Length of each random walk session.')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for each training sample.')
    parser.add_argument('--n_walks_per_node', type=int, default=3, help='Number of walks to start from each node.')
    parser.add_argument('--sample_node_prob', type=float, default=0.05, help='Probability of sampling a node.')
    
    parser.add_argument('--pretrain', type=str, default=None, help='Path to the pre-trained model.')
    
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')

    args = parser.parse_args()
    seed_everything(args.seed)

    return args


def get_vaildate_args():
    parser = argparse.ArgumentParser(description='Validate model or baseline.')

    # classifier train
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the classifier.')
    parser.add_argument('--batch_size', type=int, default=int(2**12), help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run the training.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for parallel processing.')
    
    # validate options
    parser.add_argument('--model_type', type=str, default='scibert', help='Path to the pre-trained model.')
    parser.add_argument('--pretrain', type=str, default=None, help='Path to the pre-trained model.')
    
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    
    args = parser.parse_args()
    seed_everything(args.seed)

    return args


def seed_everything(seed):
    """
    Seed all random number generators for reproducibility.
    
    Args:
    - seed (int): The seed value to use.
    """
    # Seed Python random module
    random.seed(seed)
    
    # Seed NumPy
    np.random.seed(seed)
    
    # Seed PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # For CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable benchmark mode for optimal performance
