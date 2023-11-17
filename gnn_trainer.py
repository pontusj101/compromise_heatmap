from data_producer import produce_training_data
import matplotlib.pyplot as plt

data_series = produce_training_data()

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model
model = GCN(num_node_features=26, num_classes=2) # 26 features, 2 classes (compromised/not compromised)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# DataLoader
data_loader = DataLoader(data_series, batch_size=1, shuffle=True)

# Training loop
loss_values = []
for epoch in range(200):  # number of epochs
    epoch_loss = 0.0
    for batch in data_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(data_loader)
    print(f'Epoch {epoch} loss: {epoch_loss}')
    loss_values.append(epoch_loss)

plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()