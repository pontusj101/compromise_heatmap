import io
import networkx as nx
import logging
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import torch
import math
import json
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
from threading import Thread, Event
from io import BytesIO
import base64


from google.cloud import storage
from .predictor import Predictor
from .bucket_manager import BucketManager
from .gnn import GNN_LSTM
from maltoolbox.model import Model
from maltoolbox.attackgraph import AttackGraph
from maltoolbox.language import LanguageGraph, LanguageClassesFactory

logger = logging.getLogger(__name__)

class Animator:
    def __init__(self, domain_lang, animation_sequence_filename,
                 predictor_type, predictor_filename, bucket_manager,
                 hide_prediction=False, hide_state=False):

        self.predictor_type = predictor_type
        self.predictor_filename = predictor_filename

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

        model = Model.load_from_file("../mal-petting-zoo-simulator/tests/example_model.yml",
                               lang_classes_factory)
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
        if num % 50 == 0 or True:
            logger.info(f'Animating step {num}/{len(self.snapshot_sequence)}. Time: {time.time() - self.start_time:.2f}s.')
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

        return ax


    def create_animation(self, frames_per_second=25):
        logger.info(f'Animating {len(self.snapshot_sequence)} frames of {self.predictor_type} predictor {self.predictor_filename} on {self.animation_sequence_filename}.')

        predictor = Predictor(self.predictor_type, self.predictor_filename,
                              self.bucket_manager)

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

        logger.info(f'Showing both prediction and state.')
        self.hide_state = False
        self.hide_prediction = False
        ani_all = animation.FuncAnimation(fig, self.update_graph, frames=len(self.snapshot_sequence),
                                    fargs=(pos, ax, probabilities), interval=int(1000/frames_per_second))
        ani_all.save('network_animation_state_and_pred.mp4', writer='ffmpeg', fps=frames_per_second)

        logger.info(f'Showing only prediction.')
        self.hide_state = True
        self.hide_prediction = False
        ani_hide_state = animation.FuncAnimation(fig, self.update_graph, frames=len(self.snapshot_sequence),
                                    fargs=(pos, ax, probabilities), interval=int(1000/frames_per_second))
        ani_hide_state.save('network_animation_pred.mp4', writer='ffmpeg', fps=frames_per_second)
#
        logger.info(f'Showing only state.')
        self.hide_state = False
        self.hide_prediction = True
        ani_hide_state = animation.FuncAnimation(fig, self.update_graph, frames=len(self.snapshot_sequence),
                                    fargs=(pos, ax, probabilities), interval=int(1000/frames_per_second))
        ani_hide_state.save('network_animation_state.mp4', writer='ffmpeg', fps=frames_per_second)
#
        logger.info(f'Showing neither prediction nor state.')
        self.hide_state = True
        self.hide_prediction = True
        ani_hide_state = animation.FuncAnimation(fig, self.update_graph, frames=len(self.snapshot_sequence),
                                    fargs=(pos, ax, probabilities), interval=int(1000/frames_per_second))
        ani_hide_state.save('network_animation_none.mp4', writer='ffmpeg', fps=frames_per_second)


        # Optional: Close the plot to prevent display issues in some environments
        plt.close(fig)

class GraphWebApp(Animator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'secret!'
        self.socketio = SocketIO(self.app)

        # Register routes
        self.app.route('/')(self.index)

        # Register SocketIO events
        self.socketio.on_event('connect', self.test_connect, namespace='/test')
        self.socketio.on_event('disconnect', self.test_disconnect, namespace='/test')

        self.thread = Thread()
        self.thread_stop_event = Event()

    @staticmethod
    def convert_numpy_int64(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, list):
            return [GraphWebApp.convert_numpy_int64(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: GraphWebApp.convert_numpy_int64(v) for k, v in obj.items()}
        else:
            return obj

    def serialize_graph(self, G):
        data = nx.readwrite.json_graph.node_link_data(G)
        data = GraphWebApp.convert_numpy_int64(data)
        return json.dumps(data)

    def update_graph(self, num, pos, ax, probabilities):
        ax = super().update_graph(num, pos, ax, probabilities)

        fig = ax.figure  # Get the Figure object from 'ax'

        # Create a BytesIO buffer to save the image
        buffer = BytesIO()
        # Save the figure to the buffer
        fig.savefig(buffer, format='png', bbox_inches='tight')  # 'bbox_inches' ensures the whole graph is captured
        plt.close(fig)  # Close the figure to free up memory
        # Seek to the beginning of the buffer
        buffer.seek(0)
        # Encode the image in buffer to Base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return image_base64

    def send_graph_update(self):
        G = self.create_graph(0)

        logger.info('predicting probabilities')
        predictor = Predictor(self.predictor_type, self.predictor_filename,
                              self.bucket_manager)

        logger.info('predicted probabilities')

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


        frame=0
        while not self.thread_stop_event.is_set():
            image = self.update_graph(frame, pos, ax, probabilities)
            self.socketio.emit('update_image', {'image': image}, namespace='/test')
            # time.sleep(1)
            frame += 1
            if frame == len(self.snapshot_sequence):
                break

    def index(self):
        return render_template('index.html')

    def test_connect(self):
        print('Client connected')

        if not self.thread.is_alive():
            print("Starting Thread")
            self.thread = Thread(target=self.send_graph_update)
            self.thread.start()

    def test_disconnect(self):
        print('Client disconnected')
        self.thread_stop_event.set()


class Demo(GraphWebApp):
    def send_graph_update(self):
        G = self.create_graph(0)

        logger.info('predicting probabilities')
        predictor = Predictor(self.predictor_type, self.predictor_filename,
                              self.bucket_manager)

        logger.info('predicted probabilities')

        snapshot_sequence_log_window_size = self.snapshot_sequence[0].x.shape[1]

        if isinstance(predictor.model, GNN_LSTM):
            model_log_window = 2
        else:
            model_log_window = predictor.model.layers[0].in_channels
        if model_log_window < snapshot_sequence_log_window_size:
            for snapshot in self.snapshot_sequence:
                snapshot.x = snapshot.x[:, :model_log_window]



        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate layout once
        G_initial = self.create_graph(num=0)
        pos = nx.spring_layout(G_initial)  # You can use other layouts as well
        pos = nx.kamada_kawai_layout(G_initial)


        frame=0
        while not self.thread_stop_event.is_set():
            probabilities = predictor.predict_sequence(self.snapshot_sequence[frame:frame+1:1])
            image = self.update_graph(0, pos, ax, probabilities)
            self.socketio.emit('update_image', {'image': image}, namespace='/test')
            # time.sleep(1)
            frame += 1
            if frame == len(self.snapshot_sequence):
                break
            input('press enter')
            # time.sleep(2)

