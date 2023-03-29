import torch
import torch.nn as nn

# from src.step4_gcn_layer_model import GraphConvLayer
from step4_gcn_layer_model import GraphConvLayer


class GCN_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gru_input, gru_hidden_dim):
        super(GCN_GRU, self).__init__()
        self.conv1 = GraphConvLayer(input_dim, hidden_dim)  # First graph convolutional layer
        self.conv2 = GraphConvLayer(hidden_dim, output_dim)  # Second graph convolutional layer
        self.gru = nn.GRU(gru_input, gru_hidden_dim, batch_first=True)  # GRU layer

    def forward(self, adj_matrix, attr_matrix):
        # Perform the first graph convolutional layer
        hidden1 = self.conv1(adj_matrix, attr_matrix)

        # Perform the second graph convolutional layer. View() combines the last two dimensions for the GRU layer
        hidden2 = self.conv2(adj_matrix, hidden1).view(1, 168, 91)
        
        # Perform the GRU layer
        gru_out, _ = self.gru(hidden2)

        # Hyperbolic Tangent activation function
        out = gru_out.squeeze(0)
        return out