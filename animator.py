import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from simulator import produce_training_data_parallel


def create_graph(snapshot):
    G = nx.Graph()
    edge_index = snapshot.edge_index.numpy()
    node_status = snapshot.y.numpy()
    node_type = snapshot.x[:, 0].numpy()  # Assuming 1 for host, 0 for credentials

    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])

    for node, status in enumerate(node_status):
        G.nodes[node]['status'] = status
        G.nodes[node]['type'] = node_type[node]

    return G

def update_graph(num, snapshots, pos, ax):
    ax.clear()
    snapshot = snapshots[num]
    G = create_graph(snapshot)

    # Separate nodes by type
    host_nodes = [node for node, attr in G.nodes(data=True) if attr['type'] == 1]
    credential_nodes = [node for node, attr in G.nodes(data=True) if attr['type'] == 0]

    # Update node colors based on their status
    color_map_host = ['red' if G.nodes[node]['status'] == 1 else 'green' for node in host_nodes]
    color_map_credential = ['red' if G.nodes[node]['status'] == 1 else 'green' for node in credential_nodes]

    # Draw nodes separately according to their type
    nx.draw_networkx_nodes(G, pos, nodelist=host_nodes, node_color=color_map_host, node_shape='s', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=credential_nodes, node_color=color_map_credential, node_shape='o', ax=ax)

    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)

    ax.set_title(f"Step {num}")

def create_animation(snapshot_sequence):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate layout once
    G_initial = create_graph(snapshot_sequence[0])
    pos = nx.spring_layout(G_initial)  # You can use other layouts as well

    ani = animation.FuncAnimation(fig, update_graph, frames=len(snapshot_sequence), 
                                  fargs=(snapshot_sequence, pos, ax), interval=1000)
    ani.save('network_animation.gif', writer='pillow', fps=25)


n_completely_compromised, snapshot_sequence = produce_training_data_parallel(use_saved_data=False, 
                                                    n_simulations=1, 
                                                    log_window=16, 
                                                    game_time=500,
                                                    max_start_time_step=266, 
                                                    max_log_steps_after_total_compromise=8,
                                                    graph_size='medium', 
                                                    rddl_path='content/', 
                                                    random_cyber_agent_seed=None)

create_animation(snapshot_sequence)