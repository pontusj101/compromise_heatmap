import logging
import time
import torch
import numpy as np
from predictor import Predictor

class Explorer:
    def __init__(self, predictor_type, predictor_filename):
        predictor = Predictor(predictor_type, predictor_filename)


    def evaluate_sequence(self):
        combined_features = torch.cat((graph_index.node_features, log_feature_vectors), dim=1)
        snapshot = Data(x=combined_features, edge_index=graph_index.edge_index, edge_type=graph_index.edge_type, y=labels)

        predicted_labels = predictor.predict(snapshot)

    