import torch
import torch.nn as nn


class GraphConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvLayer, self).__init__()
        self.dropout = nn.Dropout(p=0.025)
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
        # out = self.dropout(out)
        return out
