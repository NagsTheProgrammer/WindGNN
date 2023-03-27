import torch.nn as nn

class GCNGRUModel(nn.Module):
    def __init__(self, gcn_model, gru_model):
        super(GCNGRUModel, self).__init__()
        self.gcn_model = gcn_model
        self.gru_model = gru_model

    def forward(self, x, edge_index):
        x = self.gcn_model(x, edge_index)
        x = x.unsqueeze(0)  # Add a sequence dimension to feed into the GRU model
        x = self.gru_model(x)
        return x