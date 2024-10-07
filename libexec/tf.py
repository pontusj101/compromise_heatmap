import logging
import logging as l
import time
from pathlib import Path

import hypertune
import tensorflow as tf
import tensorflow_gnn as tfgnn
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow_gnn.models import mt_albis

l = logging.getLogger(__name__)


class GCN(tf.keras.Model):
    pass


class GAT(tf.keras.Model):
    def __init__(self, layer_sizes, heads, num_edge_types, edge_embedding_dim):
        super().__init__()

        self.ls = layer_sizes
        self.h = heads
        self.net = num_edge_types
        self.eed = edge_embedding_dim

        print(self.ls)
        print(self.h)
        print(self.net)
        print(self.eed)

        self.edge_type_embedding = tf.keras.layers.Embedding(
            num_edge_types, edge_embedding_dim
        )

        self.layers_ = []

        in_channels = layer_sizes[0]

        for i in range(len(layer_sizes) - 1):
            out_channels = layer_sizes[i + 1]

            self.layers_.append(
                mt_albis.MtAlbisGraphUpdate(
                    units=out_channels,
                    message_dim=edge_embedding_dim,
                    attention_type="multi_head",
                    normalization_type="layer",
                    attention_num_heads=heads[i],
                    next_state_type="dense",
                    state_dropout_rate=0.2,
                    l2_regularization=1e-5,
                    receiver_tag=tfgnn.TARGET,
                )
            )
            in_channels = out_channels

    def call(self, sample):
        gnn_outs = []
        for snapshot in sample:
            # Extract the 'graph_tensor' from the dictionary
            graph_tensor = snapshot["graph_tensor"]

            # Pass the GraphTensor to the layers
            for i, layer in enumerate(self.layers_):
                graph_tensor = layer(graph_tensor)

            gnn_outs.append(graph_tensor.node_sets["nodes"].features["hidden_state"])

        return tf.transpose(tf.stack(gnn_outs, axis=0), perm=[1, 0, 2])


def calculate_loss(logits, target_labels, minority_weight):
    loss = 0
    for t in range(logits.shape[0]):
        timestep_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_labels[t, :],
            logits=tf.nn.log_softmax(logits[t, :, :], axis=-1),
        )
        loss += timestep_loss

    class_weights = tf.constant([1.0, minority_weight], dtype=tf.float32)

    weighted_loss = tf.reduce_mean(loss * tf.gather(class_weights, target_labels))

    return weighted_loss / tf.cast(logits.shape[0], tf.float32)


def train(
    model, training_samples, validation_samples, epochs, learning_rate, minority_weight
):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)


    train_loss_values = []
    validation_loss_values = []
    l.info("Starting training with %d samples.", len(training_samples))
    l.info("Validating with %d samples.", len(validation_samples))

    for epoch in range(epochs):
        start_time = time.time()
        training_loss = 0.0

        for i, sample in enumerate(training_samples):
            with tf.GradientTape() as tape:
                logits = model(sample)
                targets = tf.stack([snapshot["y"] for snapshot in sample], axis=0)
                targets = tf.transpose(targets, perm=[1, 0])

                loss = calculate_loss(logits, targets, minority_weight)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )
            training_loss += loss.numpy()  # Convert loss to float

            l.debug(
                f"Epoch {epoch}, Batch {i + 1}/{len(training_samples)}, Training Loss: {loss.numpy():.4f}."
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
            f"Epoch {epoch}: F1: {f1:.4f}. Precision: {precision:.4f}. Recall: {recall:.4f}. "
            f"Training Loss: {training_loss:.4f}. Validation Loss: {validation_loss:.4f}. "
            f"Time: {end_time - start_time:.4f}s."
        )

    return model


def convert_sample_to_tensors(sample):
    tf_sample = []

    for snapshot in sample:
        x = tf.convert_to_tensor(
            snapshot["x"], dtype=tf.float32
        )  # Convert node features
        edge_index = tf.convert_to_tensor(
            snapshot["edge_index"], dtype=tf.int64
        )  # Convert edge indices
        edge_type = tf.convert_to_tensor(
            snapshot["edge_type"], dtype=tf.int64
        )  # Convert edge types
        y = tf.convert_to_tensor(snapshot["y"], dtype=tf.int64)  # Convert labels

        # Create GraphTensor
        node_size = x.shape[0]
        edge_size = edge_index.shape[1]

        graph_tensor = tfgnn.GraphTensor.from_pieces(
            node_sets={
                "nodes": tfgnn.NodeSet.from_fields(
                    sizes=[node_size], features={"hidden_state": x}
                )
            },
            edge_sets={
                "edges": tfgnn.EdgeSet.from_fields(
                    sizes=[edge_size],
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("nodes", edge_index[0]), target=("nodes", edge_index[1])
                    ),
                    features={"edge_type": edge_type},
                )
            },
        )

        # Add the processed data (GraphTensor and labels) to the sample list
        tf_sample.append(
            {
                "graph_tensor": graph_tensor,
                "y": y,  # Target labels for training or evaluation
            }
        )

    return tf_sample


def evaluate_model(model, validation_samples, minority_weight):
    total_loss = 0
    all_predicted_labels = []
    all_true_labels = []

    # We don't need model.eval() in TensorFlow, but we stop gradient computation using tf.GradientTape
    for sequence in validation_samples:
        # Forward pass without gradient tracking
        logits = model(sequence)

        # Stack the true labels (y) for each snapshot in the sequence
        targets = tf.stack(
            [snapshot["y"] for snapshot in sequence], axis=0
        )  # Assume 'y' contains the labels
        targets = tf.transpose(targets, perm=[1, 0])  # Align targets to logits' shape

        # Compute the loss
        loss = calculate_loss(logits, targets, minority_weight)
        total_loss += loss.numpy()

        # Compute probabilities using softmax
        probabilities = tf.nn.softmax(logits, axis=-1)

        # Get predicted labels (argmax over the class dimension)
        predicted_labels = tf.argmax(probabilities, axis=-1)

        all_predicted_labels.append(
            tf.reshape(predicted_labels, [-1])
        )  # Flatten the predictions
        all_true_labels.append(tf.reshape(targets, [-1]))  # Flatten the true labels

    # Convert predictions and true labels to numpy arrays
    all_predicted_labels = tf.concat(all_predicted_labels, axis=0).numpy()
    all_true_labels = tf.concat(all_true_labels, axis=0).numpy()

    return total_loss / len(validation_samples), all_predicted_labels, all_true_labels


def predict(model, snapshot):
    logits = model(snapshot)
    probabilities = tf.nn.softmax(logits, axis=1)
    return probabilities[:, 1]


def predict_sequence(model, sequence):
    logits = model(sequence)
    probabilities = tf.nn.softmax(logits, axis=-1)
    return probabilities[:, :, 1]


def save_model(model, path: Path):
    path = path.with_suffix(path.suffix + ".tf")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(path)
    return path


def load_model(path: Path):
    path = path.with_suffix(path.suffix + ".tf")
    filename = path.name
    parts = filename.split("_")

    hidden_layers = [int(layer) for layer in parts[0].split("-")[1].split(",")]
    heads = int(parts[3].split("-")[1])
    num_edge_types = int(parts[4].split("-")[1])
    edge_embedding_dim = int(parts[5].split("-")[1])

    model = GAT(
        layer_sizes=hidden_layers,
        heads=[heads] * (len(hidden_layers) - 2) + [1],
        num_edge_types=num_edge_types,
        edge_embedding_dim=edge_embedding_dim,
    )

    model.load_weights(path.as_posix())

    return model
