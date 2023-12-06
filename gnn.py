import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, RGCNConv, Sequential


class GCN(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(GCNConv(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:  # Apply ReLU and Dropout to all but the last layer
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        
        # return F.log_softmax(x, dim=1)
        return x


class RGCN(torch.nn.Module):
    def __init__(self, layer_sizes, num_relations):
        super(RGCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        
        # Create RGCN layers based on layer sizes
        for i in range(len(layer_sizes) - 1):
            self.layers.append(RGCNConv(layer_sizes[i], layer_sizes[i+1], num_relations=num_relations))

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type)
            if i < len(self.layers) - 1:  # Apply ReLU and Dropout to all but the last layer
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        return x


class GIN(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(GIN, self).__init__()

        self.layers = Sequential('x, edge_index', [
            (GINConv(Sequential('x', [
                (torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]), 'x -> x'),
                (torch.nn.ReLU(inplace=True), 'x -> x'),
                (torch.nn.BatchNorm1d(layer_sizes[i + 1]), 'x -> x')
            ])), 'x, edge_index -> x')
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.layers(x, edge_index)
        return x


