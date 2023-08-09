import torch.nn as nn

from step5_gcn_layer_model import GraphConvLayer


class GCN_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gru_input, gru_hidden_dim):
        super(GCN_GRU, self).__init__()
        self.conv1 = GraphConvLayer(input_dim, hidden_dim)  # First graph convolutional layer
        self.conv2 = GraphConvLayer(hidden_dim, output_dim)  # Second graph convolutional layer
        self.gru = nn.GRU(gru_input, gru_hidden_dim, batch_first=True)  # GRU layer

    def forward(self, adj_matrix, attr_matrix):
        # Perform the first graph convolutional layer
        num_seq = attr_matrix.size(1)
        flat = attr_matrix.size(2) * 15 # 15 for OG paper, 2 for wind only, 4 for wind and direction
        hidden1 = self.conv1(adj_matrix, attr_matrix)

        # Perform the second graph convolutional layer. View() combines the last two dimensions for the GRU layer
        hidden2 = self.conv2(adj_matrix, hidden1).view(1, num_seq, flat)

        # Perform the GRU layer
        gru_out, _ = self.gru(hidden2)

        # Hyperbolic Tangent activation function
        out = gru_out.squeeze(0)
        return out
