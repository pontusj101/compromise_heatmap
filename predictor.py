import torch
import pickle

class Predictor:
    def __init__(self, predictor_type, file_name):
        self.predictor_type = predictor_type
        if predictor_type == 'gnn':
            self.model = torch.load(file_name)
            self.model.eval()
        elif predictor_type == 'tabular':
            with open(file_name, 'rb') as file:
                indexed_snapshot_sequence = pickle.load(file)
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
        return torch.round(torch.nan_to_num(hits/count))

    def predict(self, snapshot):
        if self.predictor_type == 'gnn':
            out = self.model(snapshot)
            return out.max(1)[1]
        elif self.predictor_type == 'tabular':
            return self.frequency(snapshot.x[:, 1:])
        elif self.predictor_type == 'none':
            return torch.zeros(len(snapshot.y))
        

