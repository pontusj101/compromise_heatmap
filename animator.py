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
            
        # Create a reverse mapping from node indices to names
        self.reverse_mapping = {v: k for k, v in self.graph_index.object_mapping.items()}

    def create_graph(self, num):
        G = nx.Graph()
        snapshot = self.snapshot_sequence[num]
        edge_index = snapshot.edge_index.numpy()
        node_status = snapshot.y.numpy()
        node_type = snapshot.x[:, 0].numpy()

        for i in range(edge_index.shape[1]):
            node_u_index = edge_index[0, i]
            node_v_index = edge_index[1, i]

            # Use reverse mapping to get the correct node names
            node_u_name = self.reverse_mapping.get(node_u_index, f"Unknown_{node_u_index}")
            node_v_name = self.reverse_mapping.get(node_v_index, f"Unknown_{node_v_index}")

            G.add_edge(node_u_name, node_v_name)

        for node_index, status in enumerate(node_status):
            node_name = self.reverse_mapping.get(node_index, f"Unknown_{node_index}")
            G.nodes[node_name]['status'] = status
            G.nodes[node_name]['type'] = node_type[node_index]

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
        color_map_host = []
        for node_name in host_nodes:
            node_index = self.graph_index.object_mapping[node_name]  # Convert name to index
            status = G.nodes[node_name]['status']
            pred = prediction[node_index].item()  # Access prediction using index

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
        for node_name in credential_nodes:
            node_index = self.graph_index.object_mapping[node_name]  # Convert name to index
            status = G.nodes[node_name]['status']
            pred = prediction[node_index].item()  # Access prediction using index

            if pred == 1 and status == 0:
                color = color_yellow
            elif pred == 1 and status == 1:
                color = color_orange
            elif pred == 0 and status == 1:
                color = color_red
            else:  # pred == 0 and status == 0
                color = color_grey

            color_map_credential.append(color)

    # Node drawing
        nx.draw_networkx_nodes(G, pos, nodelist=host_nodes, node_color=color_map_host, 
                            node_shape='s', ax=ax, edgecolors='grey')  # Added grey border for host nodes
        nx.draw_networkx_nodes(G, pos, nodelist=credential_nodes, node_color=color_map_credential, 
                            node_shape='o', ax=ax, edgecolors='grey')  # Added grey border for credential nodes

        # Edge drawing with grey color
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='grey')

        # Label drawing with smaller font size
        nx.draw_networkx_labels(G, pos, ax=ax, labels={node: node for node in G.nodes()}, font_size=10)  # Reduced font size

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

