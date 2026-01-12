import networkx as nx


def create_internet2_topology():

    G = nx.DiGraph()

    # Add Core Nodes (0-7)
    core_nodes = range(8)
    G.add_nodes_from(core_nodes, type='core')

    # Add Host Nodes (h0-h7)
    host_nodes = [f'h{i}' for i in range(8)]
    G.add_nodes_from(host_nodes, type='host')

    # Define Core Links & Delays
    # Format: (Source, Destination, Delay_ms)
    core_edges = [
        (0, 1, 1.5), (1, 0, 1.5),
        (1, 2, 2.55), (2, 1, 2.55),
        (2, 5, 1.75), (5, 2, 1.75),
        (0, 7, 1.1), (7, 0, 1.1),
        (1, 6, 0.27), (6, 1, 0.27),
        (3, 4, 0.15), (4, 3, 0.15),
        (3, 7, 1.0), (7, 3, 1.0),
        (4, 5, 0.15), (5, 4, 0.15),
        (6, 7, 0.14), (7, 6, 0.14),
        (2, 3, 2.0), (3, 2, 2.0)
    ]

    # --- Define Host Access Links ---
    # Assuming negligible access delay (0.1ms)
    # Mapping based on diagram: h0-0, h1-4, h2-2, h3-3, h4-1, h5-5, h6-6, h7-7
    access_edges = [
        ('h0', 0, 0.1), (0, 'h0', 0.1),
        ('h1', 4, 0.1), (4, 'h1', 0.1),
        ('h2', 2, 0.1), (2, 'h2', 0.1),
        ('h3', 3, 0.1), (3, 'h3', 0.1),
        ('h4', 1, 0.1), (1, 'h4', 0.1),
        ('h5', 5, 0.1), (5, 'h5', 0.1),
        ('h6', 6, 0.1), (6, 'h6', 0.1),
        ('h7', 7, 0.1), (7, 'h7', 0.1)
    ]


    for u, v, w in core_edges + access_edges:
        G.add_edge(u, v, weight=w, capacity=1000)  # Bandwidth set to 1000 Mbps

    return G