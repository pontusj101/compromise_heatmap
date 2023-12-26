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

        in_channels = layer_sizes[0] 

        for i in range(len(layer_sizes) - 1):
            if heads[i] > layer_sizes[i + 1]:
                heads[i] = layer_sizes[i + 1]
            out_channels = layer_sizes[i + 1] // heads[i]
            self.layers.append(GATConv(in_channels, out_channels, heads=heads[i]))
            in_channels = out_channels * heads[i]  # Update in_channels for the next layer

    def forward(self, sequence):
        gnn_outs = []
        for snapshot in sequence:
            x, edge_index, edge_type = snapshot.x, snapshot.edge_index, snapshot.edge_type

            # Embed edge types
            edge_attr = self.edge_type_embedding(edge_type)

            for i, layer in enumerate(self.layers):
                x = layer(x, edge_index, edge_attr=edge_attr)
                if i < len(self.layers) - 1:  # Apply ReLU and Dropout to all but the last layer
                    x = F.relu(x)
                    x = F.dropout(x, training=self.training)
            gnn_outs.append(x)
        return torch.stack(gnn_outs, dim=0).transpose(0, 1)  # Shape: (batch_size, sequence_length, num_nodes * gnn_output_size)
    
    
class GNN_LSTM(torch.nn.Module):
    def __init__(self, gnn, lstm_hidden_dim, num_classes):
        super(GNN_LSTM, self).__init__()
        self.gnn = gnn  # The GNN model (GAT in this case)
        # Assume the output dimension of GNN is known or calculate dynamically
        self.lstm = nn.LSTM(input_size=gnn.layers[-1].out_channels,  # Adjust based on GAT's output
                            hidden_size=lstm_hidden_dim, 
                            batch_first=True)
        self.classifier = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, sequence, hidden_state=None):
        # Assume 'sequence' is a list of graph snapshots (each is an instance of Data)
        gnn_outs = []
        gnn_outs = self.gnn(sequence)  # Process each snapshot through GNN
        # Convert list of outputs to tensor suitable for LSTM input
        # Pass the sequence to LSTM
        lstm_out, hidden_state = self.lstm(gnn_outs, hidden_state)
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


