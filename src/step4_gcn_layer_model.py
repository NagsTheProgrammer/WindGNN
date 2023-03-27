import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
import numpy as np


class GraphConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(input_dim, output_dim))  # Weight matrix, for our problem, input_dim = output_dim
        self.bias = nn.Parameter(torch.zeros(output_dim))  # Bias vector
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, adj_matrix, attr_matrix):
        # Perform the matrix multiplication of the adjacency matrix and the attribute (feature) matrix
        adj_attr = torch.matmul(adj_matrix, attr_matrix)

        # Multiply the result by the weight matrix and add the bias vector
        out = torch.matmul(adj_attr, self.weight) + self.bias

        # Apply the ReLU activation function
        out = self.relu(out)

        return out
