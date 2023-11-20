import torch

class GraphIndex:
    def __init__(self, size='small'):
        if size == 'small':
            self.log_mapping = {
                'observed_compromise_attack___h1': 0,
                'observed_compromise_attack___h2': 1,
                'observed_crack_attack___c1': 2,
                'observed_crack_attack___c2': 3}

            self.attackstep_mapping = {
                'compromised___h1': 0,
                'compromised___h2': 1,
                'cracked___c1': 2,
                'cracked___c2': 3}

            # Node features
            self.node_features = torch.tensor([
                [1], [1],  # Host features
                [0], [0]   # Credential features
            ], dtype=torch.float)

            # Edges
            self.edge_index = torch.tensor([
                # CONNECTED Edges (Host-to-Host) and ACCESSES/STORES Edges (Credentials-to-Host, Host-to-Credentials)
                [0, 0, 0, 2, 3],
                [1, 2, 3, 0, 1]
            ], dtype=torch.long)
        elif size == 'medium':
            self.log_mapping = {
                'observed_compromise_attack___h1': 0,
                'observed_compromise_attack___h2': 1,
                'observed_compromise_attack___h3': 2,
                'observed_compromise_attack___h4': 3,
                'observed_compromise_attack___h5': 4,
                'observed_compromise_attack___h6': 5,
                'observed_crack_attack___c1': 6,
                'observed_crack_attack___c2': 7,
                'observed_crack_attack___c3': 8,
                'observed_crack_attack___c4': 9,
                'observed_crack_attack___c5': 10,
                'observed_crack_attack___c6': 11}

            self.attackstep_mapping = {
                'compromised___h1': 0,
                'compromised___h2': 1,
                'compromised___h3': 2,
                'compromised___h4': 3,
                'compromised___h5': 4,
                'compromised___h6': 5,
                'cracked___c1': 6,
                'cracked___c2': 7,
                'cracked___c3': 8,
                'cracked___c4': 9,
                'cracked___c5': 10,
                'cracked___c6': 11}

            self.node_features = torch.tensor([
                [1], [1], [1], [1], [1], [1],  # Host features
                [0], [0], [0], [0], [0], [0]   # Credential features
            ], dtype=torch.float)

            # Edges
            self.edge_index = torch.tensor([
                # CONNECTED Edges (Host-to-Host) and ACCESSES/STORES Edges (Credentials-to-Host, Host-to-Credentials)
                [0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 5, 1, 3, 3, 4],
                [1, 2, 3, 5, 4, 5, 0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 11]
            ], dtype=torch.long)
        elif size == 'large':
            print('Not implemented yet')
