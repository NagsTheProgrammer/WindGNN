import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np

# Added a combined GCN-GRU model below

class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train_gcn_model(features_df, graph, device, hidden_channels, num_classes, learning_rate, num_epochs, verbose=True):
    dataset = __dataframe_to_data_object(features_df, graph)
    model = GCNModel(num_features=dataset.num_features, hidden_channels=hidden_channels, num_classes=num_classes).to(
        device)
    data = dataset.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if verbose:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    return model



def __dataframe_to_data_object(features_df, graph):
    # Convert DataFrame to a PyTorch tensor
    features_matrix = torch.tensor(features_df.to_numpy(), dtype=torch.float)

    # Get the edge_index tensor from the graph (assuming it's a networkx graph)
    edge_index = torch.tensor(np.array(graph.edges()).T, dtype=torch.long)

    # Create a PyTorch Geometric Data object
    data = Data(x=features_matrix, edge_index=edge_index)

    return data

###################################
## Code for a two-layer GCN combined with a GRU
# Define the graph convolutional layer
class GraphConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))  # Weight matrix, for our problem, input_dim = output_dim
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
    
# Define the two-layer graph convolutional network architecture with a GRU layer
class GCN_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gru_hidden_dim):
        super(GCN_GRU, self).__init__()
        self.conv1 = GraphConvLayer(input_dim, hidden_dim)  # First graph convolutional layer
        self.conv2 = GraphConvLayer(hidden_dim, output_dim)  # Second graph convolutional layer
        self.gru = nn.GRU(output_dim, gru_hidden_dim, batch_first=True)  # GRU layer
        
    def forward(self, adj_matrix, attr_matrix):
        # Perform the first graph convolutional layer
        hidden1 = self.conv1(adj_matrix, attr_matrix)
        
        # Perform the second graph convolutional layer
        hidden2 = self.conv2(adj_matrix, hidden1)
        
        # Perform the GRU layer
        gru_out, _ = self.gru(hidden2.unsqueeze(0))
        out = gru_out.squeeze(0)
        return out