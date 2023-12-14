import random
import torch
from google.cloud import storage
from datetime import datetime
from graph_index import GraphIndex

def create_random_instance(num_hosts, num_credentials, n_init_compromised, horizon, extra_host_host_connection_ratio=0.25, rddl_path='rddl/'):

    graph_index = GraphIndex(size=None)
    # Generate hosts and credentials
    hosts = [f'h{i}' for i in range(1, num_hosts + 1)]
    credentials = [f'c{i}' for i in range(1, num_credentials + 1)]

    for i, host in enumerate(hosts):
        graph_index.object_mapping[host] = i
        graph_index.log_mapping[f'observed_compromised___{host}'] = i
        graph_index.attackstep_mapping[f'compromised___{host}'] = i
    
    for i, credential in enumerate(credentials):
        graph_index.object_mapping[credential] = i + num_hosts
        graph_index.log_mapping[f'observed_crack_attempt___{credential}'] = i + num_hosts
        graph_index.attackstep_mapping[f'cracked___{credential}'] = i + num_hosts

    # Node features
    node_features = []
    for host in hosts:
        node_features.append([1])
    for credential in credentials:
        node_features.append([0])
    graph_index.node_features = torch.tensor(node_features, dtype=torch.float)

    # Ensure each host has at least one connection
    connected_pairs = set()
    for i in range(num_hosts - 1):
        connected_pairs.add((hosts[i], hosts[i + 1]))

    # Additional random connections (optional, for more complexity)
    target_n_connections = int((1 + extra_host_host_connection_ratio)*num_hosts)
    while len(connected_pairs) < target_n_connections:
        a, b = random.sample(hosts, 2)
        if a != b and (a, b) not in connected_pairs and (b, a) not in connected_pairs and num_hosts > 4:
            connected_pairs.add((a, b))
       
    # Assign credentials to hosts
    credential_to_host = {}
    for i, credential in enumerate(credentials):
        credential_to_host[credential] = hosts[i]

    credentials_stored_on_host = {}
    for i, credential in enumerate(credentials):
        if i < 2:
            credentials_stored_on_host[credential] = hosts[0]
        else:
            credentials_stored_on_host[credential] = hosts[random.randint(0, i - 1)]

    # Edges
    source_nodes = []
    target_nodes = []
    edge_type = []

    for (h1, h2) in connected_pairs:
        source_nodes.append(hosts.index(h1))
        target_nodes.append(hosts.index(h2))
        edge_type.append(0)
        source_nodes.append(hosts.index(h2))
        target_nodes.append(hosts.index(h1))
        edge_type.append(0)

    for credential, host in credential_to_host.items():
        source_nodes.append(credentials.index(credential) + num_hosts)
        target_nodes.append(hosts.index(host))
        edge_type.append(1)
        target_nodes.append(credentials.index(credential) + num_hosts)
        source_nodes.append(hosts.index(host))
        edge_type.append(2)
        
    for credential, host in credentials_stored_on_host.items():
        source_nodes.append(hosts.index(host))
        target_nodes.append(credentials.index(credential) + num_hosts)  
        edge_type.append(3)
        target_nodes.append(hosts.index(host))
        source_nodes.append(credentials.index(credential) + num_hosts)  
        edge_type.append(4)

    # Convert lists to a PyTorch tensor in 2xN format
    graph_index.edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    graph_index.edge_type = torch.tensor(edge_type, dtype=torch.long)

    # Define non-fluents
    non_fluents = 'non-fluents simple_network {\n\tdomain = simple_compromise;\n\n\tobjects{\n\t\thost: {'
    non_fluents += ', '.join(hosts)
    non_fluents += '};\n\t\tcredentials: {'
    non_fluents += ', '.join(credentials)
    non_fluents += '};\n\t};\n\n\tnon-fluents {\n'
    for (h1, h2) in connected_pairs:
        non_fluents += f'\t\tCONNECTED({h1}, {h2});\n'
    for credential, host in credential_to_host.items():
        non_fluents += f'\t\tACCESSES({credential}, {host});\n'
        non_fluents += f'\t\tittc_crack_attempt({credential}) = {random.randint(0, 2)};\n'
    for credential, host in credentials_stored_on_host.items():
        non_fluents += f'\t\tSTORES({host}, {credential});\n'
    non_fluents += '\t};\n}'

    # Define instance
    instance = 'instance simple_network_instance {\n\tdomain = simple_compromise;\n\tnon-fluents = simple_network;\n\n\tinit-state{\n'
    
    initial_hosts = ['h1']
    for i in range(n_init_compromised - 1):
        initial_hosts.append(hosts[random.randint(1, num_hosts - 1)])
    for initial_host in initial_hosts:
        instance += f'\t\tcompromised({initial_host}) = true;\n'
    for credential in credentials:
        instance += f'\t\trttc_crack_attempt({credential}) = {random.randint(0, 2)};\n'
    for host in hosts:
        instance += f'\t\tvalue({host}) = {random.randint(0, 16)};\n'
    instance += '\t};\n\n\tmax-nondef-actions = 1;\n\thorizon = '
    instance += f'{horizon}'
    instance += ';\n\tdiscount = 1.0;\n}'

    instance_string = non_fluents + '\n\n' + instance
    return instance_string, graph_index

def graph_index(hosts, credentials, node_features, source_nodes, target_nodes, edge_type):
    graph_index = GraphIndex(size=None)
    for i, host in enumerate(hosts):
        graph_index.object_mapping[host] = i
        graph_index.log_mapping[f'observed_compromised___{host}'] = i
        graph_index.attackstep_mapping[f'compromised___{host}'] = i
    
    for i, credential in enumerate(credentials):
        graph_index.object_mapping[credential] = i + len(hosts)
        graph_index.log_mapping[f'observed_crack_attempt___{credential}'] = i + len(hosts)
        graph_index.attackstep_mapping[f'cracked___{credential}'] = i + len(hosts)
    graph_index.node_features = torch.tensor(node_features, dtype=torch.float)
    # Convert lists to a PyTorch tensor in 2xN format
    graph_index.edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    graph_index.edge_type = torch.tensor(edge_type, dtype=torch.long)
    return graph_index

def get_edges(connected_pairs, credential_to_host, credentials_stored_on_host, hosts, credentials):
        # Edges
    source_nodes = []
    target_nodes = []
    edge_type = []

    for (h1, h2) in connected_pairs:
        source_nodes.append(hosts.index(h1))
        target_nodes.append(hosts.index(h2))
        edge_type.append(0)
        source_nodes.append(hosts.index(h2))
        target_nodes.append(hosts.index(h1))
        edge_type.append(0)

    for credential, host in credential_to_host.items():
        source_nodes.append(credentials.index(credential) + len(hosts))
        target_nodes.append(hosts.index(host))
        edge_type.append(1)
        
    for credential, host in credentials_stored_on_host.items():
        source_nodes.append(hosts.index(host))
        target_nodes.append(credentials.index(credential) + len(hosts))  
        edge_type.append(2)

    return source_nodes, target_nodes, edge_type

def create_instance(
        bucket_name='gnn_rddl',
        rddl_path='rddl/',
        n_instances=1,
        min_size=8, 
        max_size=32,
        n_init_compromised=1,
        extra_host_host_connection_ratio=0.25,
        horizon=150):
    
    rddl_file_paths = []
    graph_index_file_paths = []

    for i in range(n_instances):
        num_hosts = random.randint(int(min_size/2), int(max_size/2))
        num_credentials = num_hosts
        instance_string, graph_index = create_random_instance(
            num_hosts, 
            num_credentials, 
            n_init_compromised,
            horizon=horizon, 
            extra_host_host_connection_ratio=extra_host_host_connection_ratio, 
            rddl_path=rddl_path)

        date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]

        rddl_file_path = f'{rddl_path}instance_nnodes_{2*num_hosts}_{horizon}_{date_time_str}.rddl'
        graph_index_file_path = f'{rddl_path}graph_index_nnodes_{2*num_hosts}_{horizon}_{date_time_str}.pkl'
        with open('local_instance.rddl', 'w') as f:
            f.write(instance_string)
        torch.save(graph_index, 'local_graph_index.pkl')

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        rddl_blob = bucket.blob(rddl_file_path)
        rddl_blob.upload_from_filename('local_instance.rddl')
        gi_blob = bucket.blob(graph_index_file_path)
        gi_blob.upload_from_filename('local_graph_index.pkl')


        rddl_file_paths.append(rddl_file_path)
        graph_index_file_paths.append(graph_index_file_path)

    return rddl_file_paths, graph_index_file_paths