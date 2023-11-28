import networkx as nx
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import pickle
from simulator import produce_training_data_parallel
from predictor import Predictor

class Animator:
    def __init__(self, animation_sequence_filename):
        with open(animation_sequence_filename, 'rb') as file:
            indexed_snapshot_sequence = pickle.load(file)
            self.snapshot_sequence = indexed_snapshot_sequence['snapshot_sequence']
            self.graph_index = indexed_snapshot_sequence['graph_index']



    def create_graph(self, num):
        G = nx.Graph()
        snapshot = self.snapshot_sequence[num]
        edge_index = snapshot.edge_index.numpy()
        node_status = snapshot.y.numpy()
        node_type = snapshot.x[:, 0].numpy()  # Assuming 1 for host, 0 for credentials

        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])

        for node, status in enumerate(node_status):
            G.nodes[node]['status'] = status
            G.nodes[node]['type'] = node_type[node]

        return G

    def update_graph(self, num, pos, ax, predictor):
        ax.clear()
        snapshot = self.snapshot_sequence[num]

        prediction = predictor.predict(snapshot)

        G = self.create_graph(num=num)

        # Separate nodes by type
        host_nodes = [node for node, attr in G.nodes(data=True) if attr['type'] == 1]
        credential_nodes = [node for node, attr in G.nodes(data=True) if attr['type'] == 0]

        color_grey = '#C0C0C0'
        color_orange = '#FFCC99'
        color_red = '#FF9999'
        color_yellow = '#FFFF99'

        # Update node colors based on their status
        color_map_host = []
        for node in host_nodes:
            status = G.nodes[node]['status']
            pred = prediction[node].item()  # Assuming prediction is a tensor, use .item() to get the value

            if pred == 1 and status == 0:
                color = color_red
            elif pred == 1 and status == 1:
                color = color_orange
            elif pred == 0 and status == 1:
                color = color_yellow
            else:  # pred == 0 and status == 0
                color = color_grey

            color_map_host.append(color)

        color_map_credential = []
        for node in credential_nodes:
            status = G.nodes[node]['status']
            pred = prediction[node].item()  # Assuming prediction is a tensor, use .item() to get the value

            if pred == 1 and status == 0:
                color = color_red
            elif pred == 1 and status == 1:
                color = color_orange
            elif pred == 0 and status == 1:
                color = color_yellow
            else:  # pred == 0 and status == 0
                color = color_grey

            color_map_credential.append(color)

        # Draw nodes separately according to their type
        nx.draw_networkx_nodes(G, pos, nodelist=host_nodes, node_color=color_map_host, node_shape='s', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=credential_nodes, node_color=color_map_credential, node_shape='o', ax=ax)

        # Draw edges and labels
        nx.draw_networkx_edges(G, pos, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)

        ax.set_title(f"Step {num}")
        logging.info(f'Updated graph for step {num}. Prediction: {prediction}. Truth: {snapshot.y}')


    def create_animation(self, predictor_type, predictor_filename):

        predictor = Predictor(predictor_type, predictor_filename)

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate layout once
        G_initial = self.create_graph(num=0)
        pos = nx.spring_layout(G_initial)  # You can use other layouts as well

        ani = animation.FuncAnimation(fig, self.update_graph, frames=len(self.snapshot_sequence), 
                                    fargs=(pos, ax, predictor), interval=1000)
        ani.save('network_animation.gif', writer='pillow', fps=5)

