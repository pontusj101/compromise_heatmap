import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, RGCNConv, GATConv, Sequential


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


class GAT(torch.nn.Module):
    def __init__(self, layer_sizes, heads, num_edge_types, edge_embedding_dim):
        super(GAT, self).__init__()
        self.edge_type_embedding = torch.nn.Embedding(num_edge_types, edge_embedding_dim)
        self.layers = torch.nn.ModuleList()

        # Adjust the first layer's input size to account for edge type embeddings
        in_channels = layer_sizes[0] # + edge_embedding_dim

        for i in range(len(layer_sizes) - 1):
            if heads[i] > layer_sizes[i + 1]:
                heads[i] = layer_sizes[i + 1]
            out_channels = layer_sizes[i + 1] // heads[i]
            self.layers.append(GATConv(in_channels, out_channels, heads=heads[i]))
            in_channels = out_channels * heads[i]  # Update in_channels for the next layer

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        # Embed edge types
        edge_attr = self.edge_type_embedding(edge_type)

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr=edge_attr)
            if i < len(self.layers) - 1:  # Apply ReLU and Dropout to all but the last layer
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        return x
    
    
class GNN_LSTM(torch.nn.Module):
    def __init__(self, gnn, lstm_hidden_dim, num_classes):
        super(GNN_LSTM, self).__init__()
        self.gnn = gnn
        self.lstm = nn.LSTM(input_size=gnn.output_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, data, hidden_state=None):
        gnn_out = self.gnn(data)  # Get node embeddings from GNN
        # LSTM expects input of shape (batch, seq_len, features)
        gnn_out = gnn_out.unsqueeze(1)  # Add sequence length dimension
        lstm_out, hidden_state = self.lstm(gnn_out, hidden_state)
        logits = self.classifier(lstm_out.squeeze(1))
        return logits, hidden_state


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


