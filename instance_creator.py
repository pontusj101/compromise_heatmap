def create_instance(size='medium', horizon=150, rddl_path='content/'):
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
    
    with open(rddl_path + 'instance.rddl', 'w') as f:
        f.write(instance_string)
