import io
import networkx as nx
import logging
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import torch
import math
from google.cloud import storage
from .predictor import Predictor
from .bucket_manager import BucketManager
from .gnn import GNN_LSTM
from maltoolbox.model import Model
from maltoolbox.attackgraph import AttackGraph
from maltoolbox.language import LanguageGraph, LanguageClassesFactory

class Animator:
    def __init__(self, domain_lang, animation_sequence_filename, bucket_manager,
                 hide_prediction=False, hide_state=False):
        self.animation_sequence_filename = animation_sequence_filename
        self.bucket_manager = bucket_manager
        self.hide_prediction = hide_prediction
        self.hide_state = hide_state
        self.start_time = time.time()

        with open(animation_sequence_filename, "rb") as f:
            indexed_snapshot_sequence = torch.load(f)

        # indexed_snapshot_sequence = bucket_manager.torch_load_from_bucket(animation_sequence_filename)

        lang_graph = LanguageGraph.from_mar_archive(domain_lang)
        lang_classes_factory = LanguageClassesFactory(lang_graph)
        lang_classes_factory.create_classes()

        model = Model.load_from_file("../mal-petting-zoo-simulator/tests/example_model.json", lang_classes_factory)
        model.save_to_file("tmp/model.json")

        attack_graph = AttackGraph(lang_graph, model)
        attack_graph.attach_attackers()
        self.attack_graph = attack_graph

        self.snapshot_sequence = indexed_snapshot_sequence['snapshot_sequence']
        num_nodes = len(attack_graph.nodes)
        self.figsize = (30, 30)  # Define figure size
        area_per_node = self.figsize[0] * self.figsize[1] / num_nodes
        node_width = math.sqrt(area_per_node)
        self.normal_node_size = 500 * node_width  # Define normal size
        self.enlarged_node_size = 2 * self.normal_node_size  # Define enlarged size

        # Create a reverse mapping from node indices to names
        # self.reverse_mapping = {v: k for k, v in self.graph_index.object_mapping.items()}
        # self.reverse_mapping = {i: node.id for i, node in enumerate(attack_graph.nodes)}
        # self.forward_mapping = {node.id: i for i, node in enumerate(attack_graph.nodes)}
        self.reverse_mapping = {i: i for i, node in enumerate(attack_graph.nodes)}
        self.forward_mapping = {i: i for i, node in enumerate(attack_graph.nodes)}

    def interpolate_color(self, color_start, color_end, probability):
        """ Interpolates between two colors based on a probability value. """
        color_start_rgb = mcolors.to_rgb(color_start)
        color_end_rgb = mcolors.to_rgb(color_end)
        interpolated_rgb = [start + (end - start) * probability for start, end in zip(color_start_rgb, color_end_rgb)]
        return mcolors.to_hex(interpolated_rgb)

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
            node_u_name = self.reverse_mapping[node_u_index]
            node_v_name = self.reverse_mapping[node_v_index]

            G.add_edge(node_u_name, node_v_name)

        for node_index, status in enumerate(node_status):
            node_name = self.reverse_mapping[node_index]
            G.nodes[node_name]['status'] = status
            G.nodes[node_name]['type'] = node_type[node_index]

        return G

    def process_nodes(self, nodes, snapshot, prediction):
        color_map = []
        size_map = []
        edge_colors = []
        edge_widths = []

        for node_name in nodes:
            node_index = self.forward_mapping[node_name]
            status = snapshot.y[node_index].item()
            prob = prediction[node_index].item()

            # Interpolate color based on probability
            color = 'white'
            if not self.hide_prediction:
                color = self.interpolate_color('white', 'red', prob)

            # Node size depends on latest monitored event
            node_size = self.enlarged_node_size if snapshot.x[node_index, -1] == 1 else self.normal_node_size

            # Determine node border color and width
            edge_color = 'black' if (status == 1 and not self.hide_state) else 'green'
            edge_width = 5 if (status == 1 and not self.hide_state) == 1 else 2

            color_map.append(color)
            size_map.append(node_size)
            edge_colors.append(edge_color)
            edge_widths.append(edge_width)

        return color_map, size_map, edge_colors, edge_widths

    def update_graph(self, num, pos, ax, probabilities):
        if num % 50 == 0:
            logging.info(f'Animating step {num}/{len(self.snapshot_sequence)}. Time: {time.time() - self.start_time:.2f}s.')
        ax.clear()
        snapshot = self.snapshot_sequence[num]

        prediction = probabilities[:, num]

        G = self.create_graph(num=num)

        # Separate nodes by type
        # host_nodes = [node for node, attr in G.nodes(data=True) if attr['type'] == 1]
        # credential_nodes = [node for node, attr in G.nodes(data=True) if attr['type'] == 0]
        all_nodes = [node for node, attr in G.nodes(data=True)]

        color_dark_grey = '#808080'
        color_light_grey = '#D3D3D3'
        color_red = '#FF9999'
        color_yellow = '#FFFF99'

        # Update node colors and border styles based on their status and prediction
        # color_map_host, size_map_host, edge_colors_host, edge_widths_host = self.process_nodes(host_nodes, snapshot, prediction)
        # color_map_credential, size_map_credential, edge_colors_credential, edge_widths_credential = self.process_nodes(credential_nodes, snapshot, prediction)

        color_map, size_map, edge_colors, edge_widths = self.process_nodes(all_nodes, snapshot, prediction)

        # Node drawing with specific border colors and widths
        # nx.draw_networkx_nodes(G, pos, nodelist=host_nodes, node_color=color_map_host, node_size=size_map_host,
                            # node_shape='s', ax=ax, edgecolors=edge_colors_host, linewidths=edge_widths_host)
        # nx.draw_networkx_nodes(G, pos, nodelist=credential_nodes, node_color=color_map_credential, node_size=size_map_credential,
                            # node_shape='o', ax=ax, edgecolors=edge_colors_credential, linewidths=edge_widths_credential)
        nx.draw_networkx_nodes(G, pos, nodelist=all_nodes,
                               node_color=color_map,
                               node_size=size_map,
                               node_shape='o', ax=ax, edgecolors=edge_colors,
                               linewidths=edge_widths)

        # Edge drawing with grey color
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='grey')

        # Label drawing with smaller font size
        nx.draw_networkx_labels(G, pos, ax=ax, labels={node: node for node in G.nodes()}, font_size=10)

        ax.set_title(f"Step {num}")

    def create_animation(self, predictor_type, predictor_filename, frames_per_second=25):
        logging.info(f'Animating {len(self.snapshot_sequence)} frames of {predictor_type} predictor {predictor_filename} on {self.animation_sequence_filename}.')

        predictor = Predictor(predictor_type, predictor_filename, self.bucket_manager)

        snapshot_sequence_log_window_size = self.snapshot_sequence[0].x.shape[1]
        if isinstance(predictor.model, GNN_LSTM):
            model_log_window = 2
        else:
            model_log_window = predictor.model.layers[0].in_channels
        if model_log_window < snapshot_sequence_log_window_size:
            for snapshot in self.snapshot_sequence:
                snapshot.x = snapshot.x[:, :model_log_window]


        probabilities = predictor.predict_sequence(self.snapshot_sequence)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate layout once
        G_initial = self.create_graph(num=0)
        pos = nx.spring_layout(G_initial)  # You can use other layouts as well
        pos = nx.kamada_kawai_layout(G_initial)

        logging.info(f'Showing both prediction and state.')
        self.hide_state = False
        self.hide_prediction = False
        ani_all = animation.FuncAnimation(fig, self.update_graph, frames=len(self.snapshot_sequence),
                                    fargs=(pos, ax, probabilities), interval=int(1000/frames_per_second))
        ani_all.save('network_animation_state_and_pred.mp4', writer='ffmpeg', fps=frames_per_second)

        # logging.info(f'Showing only prediction.')
        # self.hide_state = True
        # self.hide_prediction = False
        # ani_hide_state = animation.FuncAnimation(fig, self.update_graph, frames=len(self.snapshot_sequence),
        #                             fargs=(pos, ax, probabilities), interval=int(1000/frames_per_second))
        # ani_hide_state.save('network_animation_pred.mp4', writer='ffmpeg', fps=frames_per_second)

        # logging.info(f'Showing only state.')
        # self.hide_state = False
        # self.hide_prediction = True
        # ani_hide_state = animation.FuncAnimation(fig, self.update_graph, frames=len(self.snapshot_sequence),
        #                             fargs=(pos, ax, probabilities), interval=int(1000/frames_per_second))
        # ani_hide_state.save('network_animation_state.mp4', writer='ffmpeg', fps=frames_per_second)

        # logging.info(f'Showing neither prediction nor state.')
        # self.hide_state = True
        # self.hide_prediction = True
        # ani_hide_state = animation.FuncAnimation(fig, self.update_graph, frames=len(self.snapshot_sequence),
        #                             fargs=(pos, ax, probabilities), interval=int(1000/frames_per_second))
        # ani_hide_state.save('network_animation_none.mp4', writer='ffmpeg', fps=frames_per_second)


        # Optional: Close the plot to prevent display issues in some environments
        plt.close(fig)


