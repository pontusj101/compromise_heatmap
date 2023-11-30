import networkx as nx
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import pickle
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

        color_dark_grey = '#C0C0C0'
        color_light_grey = '#D3D3D3'
        color_red = '#FF9999'
        color_yellow = '#FFFF99'

        # Update node colors and border styles based on their status and prediction
        color_map_host = []
        edge_colors_host = []
        edge_widths_host = []
        for node_name in host_nodes:
            node_index = self.graph_index.object_mapping[node_name]  # Convert name to index
            status = G.nodes[node_name]['status']
            pred = prediction[node_index].item()  # Access prediction using index

            # Determine node color
            color = color_yellow if pred == 1 else color_light_grey

            # Determine node border color and width
            edge_color = color_red if status == 1 else color_dark_grey
            edge_width = 2 if status == 1 else 1

            color_map_host.append(color)
            edge_colors_host.append(edge_color)
            edge_widths_host.append(edge_width)

        color_map_credential = []
        edge_colors_credential = []
        edge_widths_credential = []
        for node_name in credential_nodes:
            node_index = self.graph_index.object_mapping[node_name]  # Convert name to index
            status = G.nodes[node_name]['status']
            pred = prediction[node_index].item()  # Access prediction using index

            # Determine node color
            color = color_yellow if pred == 1 else color_dark_grey

            # Determine node border color and width
            edge_color = color_red if status == 1 else 'grey'
            edge_width = 2 if status == 1 else 1

            color_map_credential.append(color)
            edge_colors_credential.append(edge_color)
            edge_widths_credential.append(edge_width)

        # Node drawing with specific border colors and widths
        nx.draw_networkx_nodes(G, pos, nodelist=host_nodes, node_color=color_map_host, 
                            node_shape='s', ax=ax, edgecolors=edge_colors_host, linewidths=edge_widths_host)
        nx.draw_networkx_nodes(G, pos, nodelist=credential_nodes, node_color=color_map_credential, 
                            node_shape='o', ax=ax, edgecolors=edge_colors_credential, linewidths=edge_widths_credential)

        # Edge drawing with grey color
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='grey')

        # Label drawing with smaller font size
        nx.draw_networkx_labels(G, pos, ax=ax, labels={node: node for node in G.nodes()}, font_size=10)

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

