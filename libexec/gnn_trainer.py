import logging
import os
import random
import time

import hypertune
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

from .gnn import GAT, GCN

logger = logging.getLogger(__name__)


def print_results(
    methods, snapshot_sequence, test_true_labels, test_predicted_labels, start_time
):
    true_positives = np.sum(
        np.logical_and(test_predicted_labels == 1, test_true_labels == 1)
    )
    false_positives = np.sum(
        np.logical_and(test_predicted_labels == 1, test_true_labels == 0)
    )
    false_negatives = np.sum(
        np.logical_and(test_predicted_labels == 0, test_true_labels == 1)
    )
    true_negatives = np.sum(
        np.logical_and(test_predicted_labels == 0, test_true_labels == 0)
    )
    logger.info(f"{methods} training completed. Time: {time.time() - start_time:.2f}s.")
    logger.debug(f"Test: Predicted Labels: \n{test_predicted_labels}")
    logger.debug(f"Test: True Labels: \n{test_true_labels}")
    logger.info(
        f"{methods}. Test: True Positives: {true_positives}, False Positives: {false_positives}, False Negatives: {false_negatives}, True Negatives: {true_negatives}."
    )
    precision = precision_score(
        test_true_labels, test_predicted_labels, average="binary", zero_division=0
    )
    recall = recall_score(
        test_true_labels, test_predicted_labels, average="binary", zero_division=0
    )
    f1 = f1_score(
        test_true_labels, test_predicted_labels, average="binary", zero_division=0
    )
    logger.warning(
        f"{methods}. Test: F1 Score: {f1:.2f}. Precision: {precision:.2f}, Recall: {recall:.2f}. {len(snapshot_sequence)} snapshots."
    )


def save_model_to_bucket(
    bucket_manager,
    model_path,
    model,
    gnn_type,
    training_sequence_filenames,
    hidden_layers,
    lstm_hidden_dim,
    learning_rate,
    batch_size,
    snapshot_sequence_length,
    date_time_str,
):
    model_file_name = model_filename(
        model_path,
        gnn_type,
        training_sequence_filenames,
        hidden_layers,
        lstm_hidden_dim,
        learning_rate,
        batch_size,
        snapshot_sequence_length,
        date_time_str,
    )
    bucket_manager.torch_save_to_bucket(model, model_file_name)
    return model_file_name


def model_filename(
    model_path,
    gnn_type,
    training_sequence_filenames,
    hidden_layers,
    lstm_hidden_dim,
    learning_rate,
    batch_size,
    snapshot_sequence_length,
    date_time_str,
):
    snapshot_name = os.path.commonprefix(training_sequence_filenames).replace(
        "training_sequences/", ""
    )
    filename_root = f"{gnn_type}/{snapshot_name}_hl{hidden_layers}lstm_{lstm_hidden_dim}_nsnpsht_{snapshot_sequence_length}_lr_{learning_rate:.4f}_bs_{batch_size}_{date_time_str}"
    filename_root = filename_root.replace("[", "_").replace("]", "_").replace(" ", "")
    model_file_name = f"{model_path}model/{filename_root}.pt"
    return model_file_name


def get_num_relations(bucket_manager, training_sequence_filenames):
    first_filename = training_sequence_filenames[0]
    # first_data = bucket_manager.torch_load_from_bucket(first_filename)
    # with open('data/' + first_filename, "rb") as f:
    # first_data = torch.load(f)
    # first_snapshot = first_data['snapshot_sequence'][0]
    # num_relations = first_snapshot.num_edge_types
    return 1
    # return num_relations


def make_hidden_layers(
    n_hidden_layer_1, n_hidden_layer_2, n_hidden_layer_3, n_hidden_layer_4
):
    hidden_layers = [n_hidden_layer_1]
    if n_hidden_layer_4 > 0:
        hidden_layers = [
            n_hidden_layer_1,
            n_hidden_layer_2,
            n_hidden_layer_3,
            n_hidden_layer_4,
        ]
    elif n_hidden_layer_3 > 0:
        hidden_layers = [n_hidden_layer_1, n_hidden_layer_2, n_hidden_layer_3]
    elif n_hidden_layer_2 > 0:
        hidden_layers = [n_hidden_layer_1, n_hidden_layer_2]
    return hidden_layers


def split_snapshots(snapshot_sequence, train_share=0.8, val_share=0.2):
    """Split the snapshots into training and validation sets."""
    n_snapshots = len(snapshot_sequence)
    n_train = int(train_share * n_snapshots)
    n_val = int(val_share * n_snapshots)

    # Ensure that we have at least one snapshot in each set
    n_train = max(1, n_train)
    n_val = max(1, n_val)

    # Shuffle and split the snapshot indices
    indices = list(range(n_snapshots))
    random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]

    train_snapshots = [snapshot_sequence[i] for i in train_indices]
    val_snapshots = [snapshot_sequence[i] for i in val_indices]

    return train_snapshots, val_snapshots


def attach_forward_hook(model):
    outputs = []

    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                outputs.append((name, output[0].detach()))
                outputs.append((name, output[1][0].detach()))
                outputs.append((name, output[1][1].detach()))
            else:
                outputs.append((name, output.detach()))

        return hook

    for name, layer in model.named_modules():
        layer.register_forward_hook(get_activation(name))

    return outputs


def get_model(
    gnn_type,
    edge_embedding_dim,
    heads_per_layer,
    actual_num_features,
    num_relations,
    hidden_layers,
    lstm_hidden_dim,
):
    if gnn_type == "GCN":
        model = GCN([actual_num_features] + hidden_layers + [2])
    elif gnn_type == "GAT":
        heads = [heads_per_layer] * (len(hidden_layers)) + [
            1
        ]  # Number of attention heads in each layer
        model = GAT(
            [actual_num_features] + hidden_layers + [2],
            heads,
            num_relations,
            edge_embedding_dim,
        )
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")

    return model


def evaluate_model(model, data_loader, gnn_type, minority_weight):
    model.eval()
    total_loss = 0
    all_predicted_labels = []
    all_true_labels = []
    with torch.no_grad():
        for sequence in data_loader:
            logits = model(sequence)
            targets = torch.stack(
                [snapshot.y for snapshot in sequence], dim=0
            ).transpose(0, 1)
            loss = calculate_loss(logits, targets, minority_weight)
            total_loss += loss.item()
            probabilities = F.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(probabilities, dim=-1)
            all_predicted_labels.append(predicted_labels.cpu().numpy())
            all_true_labels.append(targets.cpu().numpy())

    all_predicted_labels = np.concatenate([l.flatten() for l in all_predicted_labels])
    all_true_labels = np.concatenate([l.flatten() for l in all_true_labels])

    return total_loss / len(data_loader), all_predicted_labels, all_true_labels


def train_gnn(
    gnn_type="GAT",
    bucket_manager=None,
    sequence_dir_path="training_sequences/",
    model_dirpath="models/",
    number_of_epochs=8,
    max_training_sequences=99999999,
    n_validation_sequences=64,
    n_uncompromised_sequences=64,
    min_nodes=0,
    max_nodes=99999999,
    min_snapshots=0,
    max_snapshots=99999999,
    log_window=99999999,
    learning_rate=0.01,
    batch_size=1,
    n_hidden_layer_1=128,
    n_hidden_layer_2=128,
    n_hidden_layer_3=0,
    n_hidden_layer_4=0,
    edge_embedding_dim=16,  # Add a parameter to set edge embedding dimension in case of GAT
    heads_per_layer=2,  # Add a parameter to set number of attention heads per layer in case of GAT
    lstm_hidden_dim=128,  # Add a parameter to set LSTM hidden dimension in case of GAT_LSTM
    minority_weight=10,
    checkpoint_interval=1,  # Add a parameter to set checkpoint interval
    checkpoint_file=None,  # Add checkpoint file parameter
    checkpoint_path="checkpoints/",
    online=False,
    fp_rate=0.1,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_values = []
    validation_loss_values = []
    logger.info(
        f"Training {gnn_type} with a log window of {log_window}, {len(training_data_loader)} graphs. Learning rate: {learning_rate}. Hidden Layers: {hidden_layers}. Validating on {len(validation_data_loader)} graphs."
    )
    for epoch in range(number_of_epochs):
        start_time = time.time()
        model.train()
        training_loss = 0.0

        for i, sequence in enumerate(training_data_loader):
            optimizer.zero_grad()
            logits = model(sequence)
            targets = torch.stack(
                [snapshot.y for snapshot in sequence], dim=0
            ).transpose(0, 1)
            loss = calculate_loss(logits, targets, minority_weight)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            logger.debug(
                f"Epoch {epoch}, Batch {i}/{len(training_data_loader)}, Processed nodes: {global_step}. Training Loss: {loss.item():.4f}."
            )

        training_loss /= len(training_data_loader)
        train_loss_values.append(training_loss)

        validation_loss, predicted_labels, true_labels = evaluate_model(
            model.to(device), validation_data_loader, gnn_type, minority_weight
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
        logger.info(
            f"Epoch {epoch}: F1: {f1:.4f}. Precision: {precision:.4f}. Recall: {recall:.4f}. Training Loss: {training_loss:.4f}. Validation Loss: {validation_loss:.4f}. {number_of_compromised_nodes} compromised nodes. {number_of_uncompromised_nodes} uncompromised nodes. Time: {end_time - start_time:.4f}s."
        )

    model = model.to("cpu")
    model_file_name = save_model_to_bucket(
        bucket_manager=bucket_manager,
        model_path=model_dirpath,
        model=model,
        gnn_type=gnn_type,
        training_sequence_filenames=training_sequence_filenames,
        hidden_layers=hidden_layers,
        lstm_hidden_dim=lstm_hidden_dim,
        learning_rate=learning_rate,
        batch_size=batch_size,
        snapshot_sequence_length=len(training_data_loader),
        date_time_str=date_time_str,
    )

    return model_file_name
