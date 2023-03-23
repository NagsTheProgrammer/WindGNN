import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def test(graph, df):
    feature_matrix = torch.tensor(df.drop(["Station ID", "Date/Time"], axis=1).values, dtype=torch.float)
    node_labels = torch.tensor(df["Wind Speed 10 m Avg. (km/h)"].values, dtype=torch.float)

    # Convert the NetworkX graph to PyG Data object
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    data = Data(x=feature_matrix, edge_index=edge_index, y=node_labels)

    # Instantiate the GCN model
    num_features = feature_matrix.shape[1]
    hidden_channels = 64
    num_classes = 1
    model = GCN(num_features, hidden_channels, num_classes)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Train the GCN model
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    # Evaluate the model
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    accuracy = correct / data.test_mask.sum().item()
    print(f"Accuracy: {accuracy:.4f}")