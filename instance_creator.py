import random
import torch
from datetime import datetime
from graph_index import GraphIndex

def create_random_instance(num_hosts, num_credentials, horizon, extra_host_host_connection_ratio=0.25, rddl_path='rddl/'):

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
    while len(connected_pairs) < int(extra_host_host_connection_ratio*num_hosts):
        a, b = random.sample(hosts, 2)
        if a != b:
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
        
    for credential, host in credentials_stored_on_host.items():
        source_nodes.append(hosts.index(host))
        target_nodes.append(credentials.index(credential) + num_hosts)  
        edge_type.append(2)

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
    initial_host = 'h1'
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


def create_mini_instance(rddl_path='rddl/'):

    # Generate hosts and credentials
    hosts = ['h1', 'h2']
    credentials = ['c1', 'c2']

    # Node features
    node_features = []
    for host in hosts:
        node_features.append([1])
    for credential in credentials:
        node_features.append([0])

    connected_pairs = set()
    connected_pairs.add(('h1', 'h2'))
       
    # Assign credentials to hosts
    credential_to_host = {}
    credentials_stored_on_host = {}
    for i, credential in enumerate(credentials):
        credential_to_host[credential] = hosts[i]
        credentials_stored_on_host[credential] = 'h1'

    source_nodes, target_nodes, edge_type = get_edges(connected_pairs, credential_to_host, credentials_stored_on_host, hosts, credentials)

    gi = graph_index(hosts, credentials, node_features, source_nodes, target_nodes, edge_type)

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
    initial_host = 'h1'
    instance += f'\t\tcompromised({initial_host}) = true;\n'
    for credential in credentials:
        instance += f'\t\trttc_crack_attempt({credential}) = {random.randint(0, 2)};\n'
    for host in hosts:
        instance += f'\t\tvalue({host}) = {random.randint(0, 16)};\n'
    instance += '\t};\n\n\tmax-nondef-actions = 1;\n\thorizon = 2;\n\tdiscount = 1.0;\n}'

    instance_string = non_fluents + '\n\n' + instance
    return instance_string, gi

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


def create_static_instance(size='medium', horizon=150, rddl_path='rddl/'):
    instance_string = '''
non-fluents simple_network {
    domain = simple_compromise;

    objects{'''
    if size == "small":
        instance_string += '''
            host: {h1, h2};
            credentials: {c1, c2};
        '''
    elif size == "medium":
        instance_string += '''
            host: {h1, h2, h3, h4, h5, h6};
            credentials: {c1, c2, c3, c4, c5, c6};
        '''
    instance_string += '''};

    non-fluents {
            CONNECTED(h1, h2);'''
    if size == "medium":
        instance_string += '''
            CONNECTED(h1, h3);
            CONNECTED(h2, h4);
            CONNECTED(h3, h6);
            CONNECTED(h4, h5);
            CONNECTED(h5, h6);''' 
    instance_string += '''
            ACCESSES(c1, h1);
            ACCESSES(c2, h2);'''
    if size == "medium":
        instance_string += '''
            ACCESSES(c3, h3);
            ACCESSES(c4, h4);
            ACCESSES(c5, h5);
            ACCESSES(c6, h6);'''
    instance_string += '''
            STORES(h1, c1);
            STORES(h1, c2);'''
    if size == "medium":
        instance_string += '''
            STORES(h6, c3);
            STORES(h2, c4);
            STORES(h4, c5);
            STORES(h5, c6);
            STORES(h4, c3);'''
    instance_string += '''
            ittc_crack_attempt(c1) = 1;
            ittc_crack_attempt(c2) = 2;'''
    if size == "medium":
        instance_string += '''
            ittc_crack_attempt(c3) = 0;
            ittc_crack_attempt(c4) = 1;
            ittc_crack_attempt(c5) = 2;
            ittc_crack_attempt(c6) = 0;'''
    instance_string += '''
    };
}

instance simple_network_instance {
    domain = simple_compromise;
    non-fluents = simple_network;

    init-state{
        compromised(h1) = true;

        rttc_crack_attempt(c1) = 1;
        rttc_crack_attempt(c2) = 2;'''
    if size == "medium":
        instance_string += '''
        rttc_crack_attempt(c3) = 0;
        rttc_crack_attempt(c4) = 1;
        rttc_crack_attempt(c5) = 2;
        rttc_crack_attempt(c6) = 0;'''
    instance_string += '''
        value(h1) = 0;
        value(h2) = 1;'''
    if size == "medium":
        instance_string += '''
        value(h3) = 2;
        value(h4) = 4;
        value(h5) = 8;
        value(h6) = 16;'''
    instance_string += '''
    };

    max-nondef-actions = 1;
    '''
    instance_string += f'horizon = {horizon};'
    instance_string += '''
    discount = 1.0;
}
'''
    return instance_string

def create_instance(instance_type='static', size='medium', horizon=150, rddl_path='rddl/'):
    if instance_type == 'static':
        instance_string = create_static_instance(size=size, horizon=horizon, rddl_path=rddl_path)
        graph_index = GraphIndex(size=size)
    elif instance_type == 'random':
        if size == 'small':
            num_hosts = 2
            num_credentials = 2
        elif size == 'medium':
            num_hosts = 6
            num_credentials = 6
        elif size == 'large':
            num_hosts = 32
            num_credentials = 32
        elif size == 'larger':
            num_hosts = 128
            num_credentials = 128
        else:  
            raise ValueError(f'Instance type {instance_type} not recognized.')
        instance_string, graph_index = create_random_instance(
            num_hosts, 
            num_credentials, 
            horizon=horizon, 
            extra_host_host_connection_ratio=0.25, 
            rddl_path=rddl_path)
    elif instance_type == 'mini':
        instance_string, graph_index = create_mini_instance(rddl_path=rddl_path)

    date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    rddl_file_path = f'{rddl_path}instance_{instance_type}_{size}_{horizon}_{date_time_str}.rddl'
    graph_index_file_path = f'{rddl_path}graph_index_{instance_type}_{size}_{horizon}_{date_time_str}.pkl'
    if instance_type == 'static':
        rddl_file_path = f'{rddl_path}instance_{instance_type}_{size}_{horizon}.rddl'
        graph_index_file_path = f'{rddl_path}graph_index_{instance_type}_{size}_{horizon}.pkl'
    elif instance_type == 'mini':
        rddl_file_path = f'{rddl_path}instance_{instance_type}.rddl'
        graph_index_file_path = f'{rddl_path}graph_index_{instance_type}.pkl'
    with open(rddl_file_path, 'w') as f:
        f.write(instance_string)
    torch.save(graph_index, graph_index_file_path)

    return rddl_file_path, graph_index_file_path