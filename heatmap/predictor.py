import io
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax
from google.cloud import storage
from .bucket_manager import BucketManager
from .gnn import GNN_LSTM


class Predictor:
    def __init__(self, predictor_type, filename, bucket_manager):
        self.predictor_type = predictor_type
        if predictor_type == 'gnn':
            self.model = bucket_manager.torch_load_from_bucket(filename)

            self.model.eval()
        elif predictor_type == 'tabular':
            indexed_snapshot_sequence = torch.load(filename)
            self.snapshot_sequence = indexed_snapshot_sequence['snapshot_sequence']

        elif predictor_type == 'none':
            pass
        else:
            raise ValueError(f'Unknown predictor type: {predictor_type}')

    def frequency(self, target_log_sequence):
        n_labels = len(self.snapshot_sequence[0].y)
        count = torch.zeros(n_labels)
        hits = torch.zeros(n_labels)
        for snapshot in self.snapshot_sequence:
            log_sequence = snapshot.x[:, 1:]
            if torch.equal(log_sequence, target_log_sequence):
                for label_index in range(n_labels):
                    count[label_index] += 1
                    labels = snapshot.y
                    if labels[label_index] == 1:
                        hits[label_index] += 1
        return torch.nan_to_num(hits/count)
        # return torch.round(torch.nan_to_num(hits/count))

    def predict(self, snapshot):
        if self.predictor_type == 'gnn':
            logits = self.model(snapshot)  # Get the raw logits from the model
            probabilities = softmax(logits, dim=1)  # Apply softmax to get probabilities
            return probabilities[:,1]  # Return the probability of the positive class
            # out = self.model(snapshot)
            # return out.max(1)[1]
        elif self.predictor_type == 'tabular':
            return self.frequency(snapshot.x[:, 1:])
        elif self.predictor_type == 'none':
            return torch.zeros(len(snapshot.y))

    def predict_sequence(self, sequence):
        hidden_state = None
        if isinstance(self.model, GNN_LSTM):
            logits, hidden_state = self.model(sequence, hidden_state)
            # hidden_state = (hidden_state[0].detach(), hidden_state[1].detach()) # Detach the hidden state to prevent backpropagation through time
        else:
            logits = self.model(sequence)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities[:,:,1]

