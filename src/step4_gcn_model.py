import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np


class GCNModel(torch.nn.Module):
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
