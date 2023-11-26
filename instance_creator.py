import random
import torch
from graph_index import GraphIndex

def create_random_instance(num_hosts, num_credentials, horizon, rddl_path='content/'):

    graph_index = GraphIndex(size=None)
    # Generate hosts and credentials
    hosts = [f'h{i}' for i in range(1, num_hosts + 1)]
    credentials = [f'c{i}' for i in range(1, num_credentials + 1)]

    for i, host in enumerate(hosts):
        graph_index.log_mapping[f'observed_compromise_attack___{host}'] = i
        graph_index.attackstep_mapping[f'compromised___{host}'] = i
    
    for i, credential in enumerate(credentials):
        graph_index.log_mapping[f'observed_crack_attack___{credential}'] = i + num_hosts
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
    while len(connected_pairs) < 2 * num_hosts:
        a, b = random.sample(hosts, 2)
        if a != b:
            connected_pairs.add((a, b))
       
    # Assign credentials to hosts
    credential_to_host = {}
    for credential in credentials:
        possible_hosts = set(hosts)
        credential_to_host[credential] = random.choice(list(possible_hosts))

    credentials_stored_on_host = {}
    for i in range(num_credentials):
        credentials_stored_on_host[credentials[i]] = hosts[0]

    # Edges
    source_nodes = []
    target_nodes = []

    for (h1, h2) in connected_pairs:
        source_nodes.append(hosts.index(h1))
        target_nodes.append(hosts.index(h2))
        source_nodes.append(hosts.index(h2))
        target_nodes.append(hosts.index(h1))

    for credential, host in credential_to_host.items():
        source_nodes.append(credentials.index(credential) + num_hosts)
        target_nodes.append(hosts.index(host))

    for credential, host in credentials_stored_on_host.items():
        source_nodes.append(hosts.index(host))
        target_nodes.append(credentials.index(credential) + num_hosts)  

    # Convert lists to a PyTorch tensor in 2xN format
    graph_index.edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

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
        non_fluents += f'\t\tittc_crack_attack({credential}) = {random.randint(0, 2)};\n'
    for credential, host in credentials_stored_on_host.items():
        non_fluents += f'\t\tSTORES({host}, {credential});\n'
    non_fluents += '\t};\n}'

    # Define instance
    instance = 'instance simple_network_instance {\n\tdomain = simple_compromise;\n\tnon-fluents = simple_network;\n\n\tinit-state{\n'
    initial_host = 'h1'
    instance += f'\t\tcompromised({initial_host}) = true;\n'
    for credential in credentials:
        instance += f'\t\trttc_crack_attack({credential}) = {random.randint(0, 2)};\n'
    for host in hosts:
        instance += f'\t\tvalue({host}) = {random.randint(0, 16)};\n'
    instance += '\t};\n\n\tmax-nondef-actions = 1;\n\thorizon = '
    instance += f'{horizon}'
    instance += ';\n\tdiscount = 1.0;\n}'

    instance_string = non_fluents + '\n\n' + instance
    return instance_string, graph_index


def create_static_instance(size='medium', horizon=150, rddl_path='content/'):
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
            ittc_crack_attack(c1) = 1;
            ittc_crack_attack(c2) = 2;'''
    if size == "medium":
        instance_string += '''
            ittc_crack_attack(c3) = 0;
            ittc_crack_attack(c4) = 1;
            ittc_crack_attack(c5) = 2;
            ittc_crack_attack(c6) = 0;'''
    instance_string += '''
    };
}

instance simple_network_instance {
    domain = simple_compromise;
    non-fluents = simple_network;

    init-state{
        compromised(h1) = true;

        rttc_crack_attack(c1) = 1;
        rttc_crack_attack(c2) = 2;'''
    if size == "medium":
        instance_string += '''
        rttc_crack_attack(c3) = 0;
        rttc_crack_attack(c4) = 1;
        rttc_crack_attack(c5) = 2;
        rttc_crack_attack(c6) = 0;'''
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

def create_instance(instance_type='static', size='medium', horizon=150, rddl_path='content/'):
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
            num_hosts = 10
            num_credentials = 10
        instance_string, graph_index = create_random_instance(num_hosts, num_credentials, horizon=horizon, rddl_path=rddl_path)

    with open(rddl_path + 'instance.rddl', 'w') as f:
        f.write(instance_string)

    return graph_index