import logging
import time
from pathlib import Path

import hypertune
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv

l = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(GCNConv(layer_sizes[i], layer_sizes[i + 1]))

        self.to(device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if (
                i < len(self.layers) - 1
            ):  # Apply ReLU and Dropout to all but the last layer
                x = F.relu(x)
                x = F.dropout(x, training=self.training)

        # return F.log_softmax(x, dim=1)
        return x


class GAT(torch.nn.Module):
    def __init__(self, layer_sizes, heads, num_edge_types, edge_embedding_dim):
        super(GAT, self).__init__()
        self.edge_type_embedding = torch.nn.Embedding(
            num_edge_types, edge_embedding_dim
        )
        self.layers = torch.nn.ModuleList()

        in_channels = layer_sizes[0]

        for i in range(len(layer_sizes) - 1):
            heads[i] = min(heads[i], layer_sizes[i + 1])
            out_channels = layer_sizes[i + 1] // heads[i]
            self.layers.append(
                GATConv(
                    in_channels,
                    out_channels,
                    edge_dim=edge_embedding_dim,
                    heads=heads[i],
                )
            )
            in_channels = (
                out_channels * heads[i]
            )  # Update in_channels for the next layer

        self.to(device)

    def forward(self, sample):
        gnn_outs = []
        for snapshot in sample:
            x, edge_index, edge_type = (
                snapshot.x,
                snapshot.edge_index,
                snapshot.edge_type,
            )
            x = x.to(torch.float)
            # edge_index = edge_index.to(torch.float)

            # Embed edge types
            edge_attr = self.edge_type_embedding(edge_type)

            for i, layer in enumerate(self.layers):
                x = layer(x, edge_index, edge_attr=edge_attr)
                if (
                    i < len(self.layers) - 1
                ):  # Apply ReLU and Dropout to all but the last layer
                    x = F.relu(x)
                    x = F.dropout(x, training=self.training)
            gnn_outs.append(x)
        return torch.stack(gnn_outs, dim=0).transpose(
            0, 1
        )  # Shape: (batch_size, sequence_length, num_nodes * gnn_output_size)


def train(
    model, training_samples, validation_samples, epochs, learning_rate, minority_weight
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_values = []
    validation_loss_values = []
    l.info("Starting training")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        training_loss = 0.0

        for i, sample in enumerate(training_samples):
            optimizer.zero_grad()
            logits = model(sample)
            targets = torch.stack([snapshot.y for snapshot in sample], dim=0).transpose(
                0, 1
            )
            loss = calculate_loss(logits, targets, minority_weight)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            l.debug(
                f"Epoch {epoch}, Batch {i + 1}/{len(training_samples)}, Training Loss: {loss.item():.4f}."
            )

        training_loss /= len(training_samples)
        train_loss_values.append(training_loss)

        validation_loss, predicted_labels, true_labels = evaluate_model(
            model, validation_samples, minority_weight
        )
        validation_loss_values.append(validation_loss)

        f1 = f1_score(true_labels, predicted_labels, average="binary", zero_division=0)
        precision = precision_score(
            true_labels, predicted_labels, average="binary", zero_division=0
        )
        recall = recall_score(
            true_labels, predicted_labels, average="binary", zero_division=0
        )

        end_time = time.time()
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="F1", metric_value=f1, global_step=epoch
        )

        l.info(
            f"Epoch {epoch}: F1: {f1:.4f}. Precision: {precision:.4f}. Recall: {recall:.4f}. Training Loss: {training_loss:.4f}. Validation Loss: {validation_loss:.4f}. Time: {end_time - start_time:.4f}s."
        )

    return model


def convert_sample_to_tensors(sample):
    torch_sample = []

    for snapshot in sample:
        x = torch.tensor(snapshot["x"], dtype=torch.float32)
        edge_index = torch.tensor(snapshot["edge_index"], dtype=torch.long)
        edge_type = torch.tensor(snapshot["edge_type"], dtype=torch.long)
        y = torch.tensor(snapshot["y"], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)

        torch_sample.append(data)

    return torch_sample


def calculate_loss(logits, target_labels, minority_weight):
    # Assume logits is of shape (batch_size, sequence_length, num_classes)
    # and target_labels is of shape (batch_size, sequence_length)
    # You might need to adapt this depending on how logits and target_labels are structured
    loss = 0
    for t in range(logits.shape[1]):  # Loop over each time step
        # loss += F.cross_entropy(F.log_softmax(logits[:, t, :], dim=1), target_labels[:,t], torch.Tensor([1, minority_weight]))
        loss += F.nll_loss(
            F.log_softmax(logits[:, t, :], dim=1),
            target_labels[:, t],
            weight=torch.Tensor([1, minority_weight]),
        )
    return loss / logits.shape[1]  # Average loss over the sequence


def evaluate_model(model, validation_samples, minority_weight):
    model.eval()
    total_loss = 0
    all_predicted_labels = []
    all_true_labels = []
    with torch.no_grad():
        for sequence in validation_samples:
            logits = model(sequence)
            targets = torch.stack(
                [snapshot.y for snapshot in sequence], dim=0
            ).transpose(0, 1)
            loss = calculate_loss(logits, targets, minority_weight)
            total_loss += loss.item()
            probabilities = F.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(probabilities, dim=-1)
            all_predicted_labels.append(predicted_labels.cpu())
            all_true_labels.append(targets.cpu())

    all_predicted_labels = torch.cat(
        [l.flatten() for l in all_predicted_labels]
    ).numpy()
    all_true_labels = torch.cat([l.flatten() for l in all_true_labels]).numpy()

    return total_loss / len(validation_samples), all_predicted_labels, all_true_labels


def predict(model, snapshot):
    logits = model(snapshot)
    probabilities = softmax(logits, dim=1)
    return probabilities[:, 1]


def predict_sequence(model, sequence):
    logits = model(sequence)
    probabilities = F.softmax(logits, dim=-1)
    return probabilities[:, :, 1]


def save_model(model, path: Path):
    path = path.with_suffix(path.suffix + ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)
    return path


def load_model(path: Path):
    path = path.with_suffix(path.suffix + ".pt")
    model = torch.load(path)
    model.eval()
    return model
