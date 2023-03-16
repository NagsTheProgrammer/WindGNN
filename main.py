import argparse
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from src.gcn_model import GCN

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()

# Load and preprocess wind speed data from CSV
preprocessed_df = load_and_process_wind_speed_dataset(r'../../data/ACISHourlyData-20200101-20221231.csv')

# Convert wind speed data into graph format
data = torch.from_numpy(preprocessed_df['wind_speed'].values).float()
edge_index = torch.from_numpy(np.array([preprocessed_df.index[:-1], preprocessed_df.index[1:]])).long()
edge_weight = torch.ones(edge_index.size(1))
data = Data(x=data, edge_index=edge_index, edge_weight=edge_weight)

# Initialize model and optimizer
model = GCN(1, args.hidden_channels, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Train model
for epoch in range(1, args.epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.mse_loss(out[:-1], data.x[1:])
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}: Loss {loss.item()}')