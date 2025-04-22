import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self, input_dim=1, out_dim=1, width=10, depth=5, activation='tanh', initialization='uniform'):
        super(MLP, self).__init__()

        # Want to ensure at least one hidden layer
        assert depth > 1
        self.depth = depth
        
        # Selecting the activation for hidden layers
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sin':
            self.activation = torch.sin
        elif activation == 'relu':
            self.activation = nn.ReLU()

        # Create a list of layers and include the first hidden layer
        MLP_list = [nn.Linear(input_dim, width)]

        # Remaining hidden layers
        for _ in range(depth - 2):
            MLP_list.append(nn.Linear(width, width))

        # Output layer 
        MLP_list.append(nn.Linear(width, out_dim))

        # Adding list of layers as modules
        self.model = nn.ModuleList(MLP_list)

        # Weights initialization
        # Solution for item 5.8.1
        def init_uniform(layer):
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -1, 1)
                if layer.bias is not None:
                    nn.init.uniform_(layer.bias, -1, 1)

        def init_normal(layer):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, mean=0.0, std=1.0)

        if initialization == 'uniform':
            self.model.apply(init_uniform)
        
        elif initialization == 'normal':
            self.model.apply(init_normal)

    # Defining forward mode of MLP model
    # Since i < depth-1, we apply this condition, activation function will run in all layers except the last one (output layer)
    def forward(self, x):

        for i, layer in enumerate(self.model):
            # Apply activation only to hidden layers
            if i < self.depth-1:
                x = self.activation(layer(x))
            else:
                x = layer(x)
 
        return x
        
    def countpar(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

# Create custom dataset Class 
class CustomDataset(Dataset):
    def __init__(self, samples):
        """ Initialize CustomDataset with paired samples.

        Args:
            samples (list of tuples): A ;ist of (x, y) pairs
            representing the dataset samples. """
        # I've decided to update to a more modern version
        self.samples = torch.tensor(samples, dtype=torch.float32)

    def __len__(self):
        """
        Returns the lenght of the dataset, i.e., the number of samples."""

        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns the sample pairs corresponding to the given list of indices.
        Args: indices (list): A list of indices to retrieve samples for.
        Returns: list: A list of (x, y) pairs corresponding to the specified indices. 
        """
        # Split the tensor into input (x) and target (y)
        x = self.samples[idx, 0]  # First column is input
        y = self.samples[idx, 1]  # Second column is target
        return x, y
