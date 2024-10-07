import logging
import math
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import torch
from malsim.sims.mal_simulator import MalSimulator
from maltoolbox.attackgraph import AttackGraph
from maltoolbox.language import LanguageClassesFactory, LanguageGraph
from maltoolbox.model import Model
from matplotlib import animation

from .gnn import GNN_LSTM
from .predictor import Predictor

logger = logging.getLogger(__name__)


class Animator:
    def __init__(
        self,
        domain_lang,
        animation_sequence_filename,
        bucket_manager,
        hide_prediction=False,
        hide_state=False,
    ):
        self.animation_sequence_filename = animation_sequence_filename
        self.bucket_manager = bucket_manager
        self.hide_prediction = hide_prediction
        self.hide_state = hide_state
        self.start_time = time.time()

        with open(animation_sequence_filename, "rb") as f:
            indexed_snapshot_sequence = torch.load(f)

        # indexed_snapshot_sequence = bucket_manager.torch_load_from_bucket(animation_sequence_filename)

        domain_lang = "../heatmap/heatmap.mal"
        try:
            lang_graph = LanguageGraph.from_mar_archive(domain_lang)
        except:
            lang_graph = LanguageGraph.from_mal_spec(domain_lang)
        lang_classes_factory = LanguageClassesFactory(lang_graph)
        # lang_classes_factory.create_classes()

        model = Model.load_from_file(
            "../heatmap/instance-model.yml", lang_classes_factory
        )
        model.save_to_file("tmp/model.json")

        attack_graph = AttackGraph(lang_graph, model)
        attack_graph.attach_attackers()
        self.attack_graph = attack_graph
        self.env = MalSimulator(lang_graph, model, attack_graph)

        self.snapshot_sequence = indexed_snapshot_sequence["snapshot_sequence"]
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
        """Interpolates between two colors based on a probability value."""
        color_start_rgb = mcolors.to_rgb(color_start)
        color_end_rgb = mcolors.to_rgb(color_end)
        # import pdb
        if probability > 0.7:
            probability = 0
        interpolated_rgb = [
            start + (end - start) * probability
            for start, end in zip(color_start_rgb, color_end_rgb, strict=False)
        ]
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
            try:
                G.nodes[node_name]["status"] = status
                G.nodes[node_name]["type"] = node_type[node_index]
            except KeyError:
                pass

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
            color = "white"
            if not self.hide_prediction:
                color = self.interpolate_color("white", "red", prob)

            # Node size depends on latest monitored event
            node_size = (
                self.enlarged_node_size
                if snapshot.x[node_index, -1] == 1
                else self.normal_node_size
            )

            # Determine node border color and width
            edge_color = "black" if (status == 1 and not self.hide_state) else "green"
            edge_width = 5 if (status == 1 and not self.hide_state) == 1 else 2

            color_map.append(color)
            size_map.append(node_size)
            edge_colors.append(edge_color)
            edge_widths.append(edge_width)

        return color_map, size_map, edge_colors, edge_widths

    def update_graph(self, num, pos, ax, probabilities):
        if num % 50 == 0:
            logger.info(
                f"Animating step {num}/{len(self.snapshot_sequence)}. Time: {time.time() - self.start_time:.2f}s."
            )
        ax.clear()
        snapshot = self.snapshot_sequence[num]

        prediction = probabilities[:, num]

        G = self.create_graph(num=num)

        # Separate nodes by type
        project_nodes = [
            node
            for node, attr in G.nodes(data=True)
            if self.attack_graph.nodes[node].asset
            and self.attack_graph.nodes[node].asset.metaconcept == "Project"
        ]
        bucket_nodes = [
            node
            for node, attr in G.nodes(data=True)
            if self.attack_graph.nodes[node].asset
            and self.attack_graph.nodes[node].asset.metaconcept == "Gcs_bucket"
        ]
        object_nodes = [
            node
            for node, attr in G.nodes(data=True)
            if self.attack_graph.nodes[node].asset
            and self.attack_graph.nodes[node].asset.metaconcept == "Gcs_object"
        ]
        sa_nodes = [
            node
            for node, attr in G.nodes(data=True)
            if self.attack_graph.nodes[node].asset
            and self.attack_graph.nodes[node].asset.metaconcept == "Service_account"
        ]
        sa_key_nodes = [
            node
            for node, attr in G.nodes(data=True)
            if self.attack_graph.nodes[node].asset
            and self.attack_graph.nodes[node].asset.metaconcept == "Service_account_key"
        ]
        vm_nodes = [
            node
            for node, attr in G.nodes(data=True)
            if self.attack_graph.nodes[node].asset
            and self.attack_graph.nodes[node].asset.metaconcept == "Gce_instance"
        ]
        # all_nodes = [node for node, attr in G.nodes(data=True)]

        color_dark_grey = "#808080"
        color_light_grey = "#D3D3D3"
        color_red = "#FF9999"
        color_yellow = "#FFFF99"

        # Update node colors and border styles based on their status and prediction
        (
            color_map_project,
            size_map_project,
            edge_colors_project,
            edge_widths_project,
        ) = self.process_nodes(project_nodes, snapshot, prediction)
        color_map_bucket, size_map_bucket, edge_colors_bucket, edge_widths_bucket = (
            self.process_nodes(bucket_nodes, snapshot, prediction)
        )
        color_map_object, size_map_object, edge_colors_object, edge_widths_object = (
            self.process_nodes(object_nodes, snapshot, prediction)
        )
        color_map_sa, size_map_sa, edge_colors_sa, edge_widths_sa = self.process_nodes(
            sa_nodes, snapshot, prediction
        )
        color_map_sa_key, size_map_sa_key, edge_colors_sa_key, edge_widths_sa_key = (
            self.process_nodes(sa_key_nodes, snapshot, prediction)
        )
        color_map_vm, size_map_vm, edge_colors_vm, edge_widths_vm = self.process_nodes(
            vm_nodes, snapshot, prediction
        )

        edge_colors_project = "black" if edge_colors_project == "black" else "blue"
        edge_colors_bucket = "black" if edge_colors_bucket == "black" else "darkgreen"
        edge_colors_object = "black" if edge_colors_object == "black" else "green"
        edge_colors_sa = "black" if edge_colors_sa == "black" else "purple"
        edge_colors_sa_key = "black" if edge_colors_sa_key == "black" else "teal"
        edge_colors_vm = "black" if edge_colors_vm == "black" else "black"

        # Node drawing with specific border colors and widths
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=project_nodes,
            node_color=color_map_project,
            node_size=size_map_project,
            node_shape="o",
            ax=ax,
            edgecolors=edge_colors_project,
            linewidths=edge_widths_project,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=bucket_nodes,
            node_color=color_map_bucket,
            node_size=size_map_bucket,
            node_shape="o",
            ax=ax,
            edgecolors=edge_colors_bucket,
            linewidths=edge_widths_bucket,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=object_nodes,
            node_color=color_map_object,
            node_size=size_map_object,
            node_shape="o",
            ax=ax,
            edgecolors=edge_colors_object,
            linewidths=edge_widths_object,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=sa_nodes,
            node_color=color_map_sa,
            node_size=size_map_sa,
            node_shape="o",
            ax=ax,
            edgecolors=edge_colors_sa,
            linewidths=edge_widths_sa,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=sa_key_nodes,
            node_color=color_map_sa_key,
            node_size=size_map_sa_key,
            node_shape="o",
            ax=ax,
            edgecolors=edge_colors_sa_key,
            linewidths=edge_widths_sa_key,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=vm_nodes,
            node_color=color_map_vm,
            node_size=size_map_vm,
            node_shape="o",
            ax=ax,
            edgecolors=edge_colors_vm,
            linewidths=edge_widths_vm,
        )

        # Edge drawing with grey color
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="grey")

        # Label drawing with smaller font size
        nx.draw_networkx_labels(
            G, pos, ax=ax, labels={node: node for node in G.nodes()}, font_size=10
        )

        ax.set_title(f"Step {num}")

    def create_animation(
        self, predictor_type, predictor_filename, frames_per_second=25
    ):
        logger.info(
            f"Animating {len(self.snapshot_sequence)} frames of {predictor_type} predictor {predictor_filename} on {self.animation_sequence_filename}."
        )

        if predictor_type is not None:
            predictor = Predictor(
                predictor_type, predictor_filename, self.bucket_manager
            )

            snapshot_sequence_log_window_size = self.snapshot_sequence[0].x.shape[1]
            if isinstance(predictor.model, GNN_LSTM):
                model_log_window = 2
            else:
                model_log_window = predictor.model.layers[0].in_channels
            if model_log_window < snapshot_sequence_log_window_size:
                for snapshot in self.snapshot_sequence:
                    snapshot.x = snapshot.x[:, :model_log_window]

            probabilities = predictor.predict_sequence(self.snapshot_sequence)

        else:
            probabilities = torch.zeros(500, 500)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate layout once
        G_initial = self.create_graph(num=0)
        # pos = nx.spring_layout(G_initial)  # You can use other layouts as well
        pos = nx.kamada_kawai_layout(G_initial)
        pos = nx.kamada_kawai_layout(G_initial, center=pos[13])

        logger.info("Showing both prediction and state.")
        self.hide_state = False
        self.hide_prediction = False
        ani_all = animation.FuncAnimation(
            fig,
            self.update_graph,
            frames=len(self.snapshot_sequence),
            # ani_all = animation.FuncAnimation(fig, self.update_graph, frames=1,
            fargs=(pos, ax, probabilities),
            interval=int(1000 / frames_per_second),
        )
        ani_all.save(
            "network_animation_state_and_pred.mp4",
            writer="ffmpeg",
            fps=frames_per_second,
        )

        logger.info("Showing only prediction.")
        self.hide_state = True
        self.hide_prediction = False
        ani_hide_state = animation.FuncAnimation(
            fig,
            self.update_graph,
            frames=len(self.snapshot_sequence),
            fargs=(pos, ax, probabilities),
            interval=int(1000 / frames_per_second),
        )
        ani_hide_state.save(
            "network_animation_pred.mp4", writer="ffmpeg", fps=frames_per_second
        )
        logger.info("Showing only state.")
        self.hide_state = False
        self.hide_prediction = True
        ani_hide_state = animation.FuncAnimation(
            fig,
            self.update_graph,
            frames=len(self.snapshot_sequence),
            fargs=(pos, ax, probabilities),
            interval=int(1000 / frames_per_second),
        )
        ani_hide_state.save(
            "network_animation_state.mp4", writer="ffmpeg", fps=frames_per_second
        )
        logger.info("Showing neither prediction nor state.")
        self.hide_state = True
        self.hide_prediction = True
        ani_hide_state = animation.FuncAnimation(
            fig,
            self.update_graph,
            frames=len(self.snapshot_sequence),
            fargs=(pos, ax, probabilities),
            interval=int(1000 / frames_per_second),
        )
        ani_hide_state.save(
            "network_animation_none.mp4", writer="ffmpeg", fps=frames_per_second
        )

        # Optional: Close the plot to prevent display issues in some environments
        plt.close(fig)
