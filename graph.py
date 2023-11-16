import torch
from torch_geometric.data import Data

node_mapping = {
    'observed_crack_attack___c1': 6,
    'observed_crack_attack___c2': 7,
    'observed_crack_attack___c3': 8,
    'observed_crack_attack___c4': 9,
    'observed_crack_attack___c5': 10,
    'observed_crack_attack___c6': 11}


# Node features
node_features = torch.tensor([
    [1], [1], [1], [1], [1], [1],  # Host features
    [0], [0], [0], [0], [0], [0]   # Credential features
])

log_traces = [[], [], ['observed_crack_attack___c2'], [], ['observed_crack_attack___c1', 'observed_crack_attack___c2'], [], ['observed_crack_attack___c5'], ['observed_crack_attack___c1', 'observed_crack_attack___c6'], [], [], [], [], ['observed_crack_attack___c2'], [], ['observed_crack_attack___c6'], ['observed_crack_attack___c3', 'observed_crack_attack___c5'], ['observed_crack_attack___c5'], ['observed_crack_attack___c2', 'observed_crack_attack___c4', 'observed_crack_attack___c5'], [], [], ['observed_crack_attack___c3', 'observed_crack_attack___c5'], [], ['observed_crack_attack___c6'], ['observed_crack_attack___c4'], ['observed_crack_attack___c3']]

N = len(node_features)
T = len(log_traces)  # Number of time steps

# Initialize feature vectors with -1
log_feature_vectors = torch.full((N, T), -1)


# Update feature vectors based on log traces
for t, log_trace in enumerate(log_traces):
    for node in range(N):
        # Check if node is observable at this time step
        if any(log_event in log_trace for log_event, idx in node_mapping.items() if idx == node):
            log_feature_vectors[node][t] = 1
        elif log_feature_vectors[node][t] == -1:
            log_feature_vectors[node][t] = 0

combined_features = torch.cat((node_features, log_feature_vectors), dim=1)

# Edges
edge_index = torch.tensor([
    # CONNECTED Edges (Host-to-Host)
    [0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 
     # ACCESSES Edges (Credentials-to-Host)
     6, 7, 8, 9, 10, 11, 
     # STORES Edges (Host-to-Credentials)
     0, 0, 5, 1, 3, 3, 4],
    [1, 2, 3, 5, 4, 5, 0, 0, 1, 2, 3, 
     0, 1, 2, 3, 4, 5, 
     6, 7, 8, 9, 10, 8, 11]
], dtype=torch.long)

# Create PyTorch Geometric data object
data = Data(x=combined_features, edge_index=edge_index)

# Print the conttent of the data object
print(data.x)
print(data.edge_index)