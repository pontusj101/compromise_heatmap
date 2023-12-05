import logging
import time
import torch
import numpy as np
from pyRDDLGym import RDDLEnv
from predictor import Predictor

class Explorer:
    def __init__(self, predictor_type, predictor_filename):
        self.predictor = Predictor(predictor_type, predictor_filename)

    def explore(self, snapshot_filepath):
        indexed_snapshot_sequence = torch.load(snapshot_filepath)
        snapshot_sequence = indexed_snapshot_sequence['snapshot_sequence']
        snapshot = snapshot_sequence[0]

        predicted_labels = self.predictor.predict(snapshot)

    