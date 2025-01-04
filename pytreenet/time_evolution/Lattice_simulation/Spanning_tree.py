import pytreenet as ptn
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx

def generate_connections(tree_dict, root):
    """
    Generate the correct connections list (parent, child, parent_leg, child_leg)
    ensuring root's parent legs start from 0, while others increment from 1.
    """
    connections = []
    parent_leg_tracker = {node: 0 for node in tree_dict.keys()}  # Track parent_leg usage

    def traverse_tree(node, is_root=False):
        # Process each child of the current node
        for child in tree_dict[node]["children"]:
            parent_leg = parent_leg_tracker[node] if is_root else parent_leg_tracker[node] + 1
            connections.append((node, child, parent_leg, 0))  # child_leg is always 0
            parent_leg_tracker[node] += 1  # Increment parent's leg tracker
            traverse_tree(child)  # Recursively process the child

    # Start traversing from the root
    traverse_tree(root, is_root=True)
    return connections

def random_spanning_tree_structure(lattice_dim_squred , bond_dim = 2):
    """
    Generate a random spanning tree using Wilson's algorithm.
    
    Parameters:
        lattice_dim_squred (int): The size of the lattice.

    Returns:
        shape_dict (dict): A dictionary mapping nodes to their tensor shape.
        connections (list): A list of connections (edges) in the spanning tree.
        tree_dict (dict): A dictionary representation of the spanning tree structure.
        root (tuple): The root node of the spanning tree.
    """
    # Generate all nodes in the lattice
    lattice_nodes = [(i, j) for i in range(lattice_dim_squred) for j in range(lattice_dim_squred)]

    # Initialize the spanning tree data structures
    tree_dict = {node: {"parent": None, "children": []} for node in lattice_nodes}
    visited = set()

    # Choose a random root node
    root = random.choice(lattice_nodes)
    visited.add(root)

    # Wilson's algorithm: Loop-erased random walk to connect unvisited nodes
    unvisited = set(lattice_nodes) - {root}
    while unvisited:
        # Start a random walk from a random unvisited node
        start = random.choice(list(unvisited))
        path = [start]

        while path[-1] not in visited:
            current = path[-1]

            # Get neighbors of the current node
            neighbors = [
                (current[0] + dx, current[1] + dy)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ]
            neighbors = [n for n in neighbors if n in lattice_nodes]

            # Randomly choose the next step in the walk
            next_node = random.choice(neighbors)
            if next_node in path:
                # Loop detected, erase the loop
                loop_start = path.index(next_node)
                path = path[:loop_start + 1]
            else:
                path.append(next_node)

        # Add the path to the spanning tree
        for i in range(len(path) - 1):
            parent, child = path[i + 1], path[i]
            tree_dict[child]["parent"] = parent
            tree_dict[parent]["children"].append(child)
            visited.add(child)
            unvisited.discard(child)

    # Generate the shape dictionary
    n_neighbour = {
        node: len(data['children']) + (1 if data['parent'] is not None else 0)
        for node, data in tree_dict.items()
    }
    shape_dict = {key: (bond_dim,) * (value) + (2,) for key, value in n_neighbour.items()}

    # Generate the connections list
    connections = generate_connections(tree_dict, root)

    return shape_dict, connections, tree_dict, root

def Visualize_tree(tree_dict, lattice_dim_squred):
    """
    Visualize the tree structure using the dictionary
    """
    G_tree = nx.Graph()  # Use an undirected graph
    for node, data in tree_dict.items():
        if data['parent'] is not None:
            G_tree.add_edge(data['parent'], node)
        for child in data['children']:
            G_tree.add_edge(node, child)

    # Define positions based on the 4x4 lattice
    lattice_nodes = [(i, j) for i in range(lattice_dim_squred) for j in range(lattice_dim_squred)]
    pos_tree = {node: (node[1], -node[0]) for node in lattice_nodes}

    # Draw the graph
    plt.figure(figsize=(3, 3))
    nx.draw(
        G_tree,
        pos_tree,
        with_labels=False,  # Do not show labels on the vertices
        node_size=50,      # Smaller circles
        node_color="black", # Black circles
        edge_color="black",
        linewidths=1,
        width=1
    )
    plt.show()

    return

def random_lattice_ttn(lattice_dim_squred, bond_dim):

    """
    Generate a random Tensor Train Network (TTN) on a lattice dimension and bond dimension.
    """
    # Generate the tree structure
    shapes, connections, tree_dict, root = random_spanning_tree_structure(lattice_dim_squred , bond_dim)

    # Generate random tensors
    nodes = {
        (i, j): ptn.random_tensor_node(shapes[(i, j)], identifier=f"Site({i},{j})") for i in range(lattice_dim_squred) for j in range(lattice_dim_squred)
    }
    # add root
    ttn = ptn.TreeTensorNetworkState()
    ttn.add_root(nodes[root][0], nodes[root][1])

    # Create the Tensor Train Network
    for (parent, child, parent_leg, child_leg) in connections:
        parent_id = f"Site({parent[0]},{parent[1]})"
        ttn.add_child_to_parent(nodes[child][0], nodes[child][1], child_leg, parent_id, parent_leg)

    local_state = [1] + [0] * (list(shapes.values())[0][-1]- 1) 
    local_state = np.array(local_state) 

    return ttn , tree_dict

def Weighted_Path_Length_Index(ttn):
    """
    Compute 1 / CV (Coefficient of Variation) for path lengths in a spanning tree.

    Args:
        ttn: A tree tensor network object with `nodes` attribute and `path_from_to` function.

    Returns:
        float: The inverse of the coefficient of variation (1 / CV) of path lengths.
    """
    # Extract all node identifiers
    node_ids = list(ttn.nodes.keys())

    # Compute all pairwise path lengths
    path_lengths = []
    for i, node1 in enumerate(node_ids):
        for j, node2 in enumerate(node_ids):
            if i < j:  # Unique pairs
                path = ttn.path_from_to(node1, node2)
                path_lengths.append(len(path) - 1)  # Length of path is number of edges
    # Compute mean and standard deviation of path lengths
    mean_length = np.mean(path_lengths)
    std_dev_length = np.std(path_lengths)

    # Compute CV and return its reciprocal
    return  100/(std_dev_length* mean_length)


def calculate_node_degree_distribution(tree_dict):
    """
    Calculates the degree distribution of nodes in the tree.

    Args:
        tree_dict (dict): A dictionary representation of the tree. 
                          Each key is a node, and its value contains 'children' and 'parent'.

    Returns:
        dict: A dictionary where keys are degrees and values are the counts of nodes with that degree.
    """
    degree_counts = {}

    for node, data in tree_dict.items():
        # Calculate the degree of the current node
        degree = len(data['children']) + (1 if data['parent'] is not None else 0)
        
        # Increment the count for this degree
        if degree not in degree_counts:
            degree_counts[degree] = 0
        degree_counts[degree] += 1

    return degree_counts


import numpy as np

def compute_scores_old(profiling_data, alpha=1, beta=1, gamma=1):
    """
    Compute scores for configurations based on Running Time, update_tree_cache_calls, and WPLI,
    using Z-score normalization and a Weighted Sum approach.
    
    Parameters:
        profiling_data (list of dicts): Profiling data with keys:
                                        'Running_Time_s', 'update_tree_cache_calls', 'WPLI'.
        alpha (float): Weight for Running Time (smaller is better).
        beta (float): Weight for update_tree_cache_calls (smaller is better).
        gamma (float): Weight for WPLI (larger is better).
    
    Returns:
        list of dicts: Updated profiling data with additional 'Z_Running_Time_s', 
                      'Z_update_tree_cache_calls', 'Z_WPLI', 'Score', and 'Normalized_Score' keys.
    """
    # Extract raw metrics
    running_times = np.array([d['Running_Time_s'] for d in profiling_data])
    cache_calls = np.array([d['update_tree_cache_calls'] for d in profiling_data])
    wpli_values = np.array([d['WPLI'] for d in profiling_data])

    # Function to compute Z-scores
    def z_score_normalize(values):
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return np.zeros_like(values)  # Avoid division by zero
        return (values - mean) / std

    # Compute Z-scores
    z_running_times = z_score_normalize(running_times)
    z_cache_calls = z_score_normalize(cache_calls)
    z_wpli = z_score_normalize(wpli_values)

    # Invert Z-scores for metrics where smaller is better
    # Running Time and update_tree_cache_calls: smaller is better
    z_running_times = -z_running_times
    z_cache_calls = -z_cache_calls
    # WPLI: larger is better (no inversion)

    # Compute weighted sum
    scores = alpha * z_running_times + beta * z_cache_calls + gamma * z_wpli

    # Normalize the composite scores between 0 and 1
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        normalized_scores = np.ones_like(scores)  # Assign 1.0 if all scores are identical
    else:
        normalized_scores = (scores - min_score) / (max_score - min_score)

    # Update profiling_data with Z-scores and scores
    scored_data = []
    for i, entry in enumerate(profiling_data):
        scored_entry = entry.copy()
        scored_entry['Z_Running_Time_s'] = z_running_times[i]
        scored_entry['Z_update_tree_cache_calls'] = z_cache_calls[i]
        scored_entry['Z_WPLI'] = z_wpli[i]
        scored_entry['Score'] = scores[i]
        scored_entry['Normalized_Score'] = normalized_scores[i]
        scored_data.append(scored_entry)

    return scored_data



def compute_scores(profiling_data, cache_calls_ncalls, alpha=1, gamma=1):
    """
    Compute scores for configurations based on Running Time and WPLI for a specific cache_calls_ncalls value.
    
    Parameters:
        profiling_data (list of dicts): Profiling data with keys:
                                        'Running_Time_s', 'update_tree_cache_calls', 'WPLI'.
        cache_calls_ncalls (int): The specific cache_calls value to process.
        alpha (float): Weight for Running_Time_s (smaller is better).
        gamma (float): Weight for WPLI (larger is better).
    
    Returns:
        list of dicts: Updated profiling data with additional 'Z_Running_Time_s', 
                      'Z_WPLI', 'Score', and 'Normalized_Score' keys for the specified category.
    """
    # Filter configurations with update_tree_cache_calls == cache_calls_ncalls
    subset = [d for d in profiling_data if d['update_tree_cache_calls'] == cache_calls_ncalls]
    
    if not subset:
        raise ValueError(f"No configurations found with update_tree_cache_calls = {cache_calls_ncalls}.")
    
    # Extract Running_Time_s and WPLI
    running_times = [d['Running_Time_s'] for d in subset]
    wpli_values = [d['WPLI'] for d in subset]
    
    # Compute means and standard deviations
    mean_rt = np.mean(running_times)
    std_rt = np.std(running_times)
    mean_wpli = np.mean(wpli_values)
    std_wpli = np.std(wpli_values)
    
    # Handle cases with zero standard deviation
    if std_rt == 0:
        z_rt = [0 for _ in running_times]
    else:
        z_rt = [(rt - mean_rt) / std_rt for rt in running_times]
    
    if std_wpli == 0:
        z_wpli = [0 for _ in wpli_values]
    else:
        z_wpli = [(wpli - mean_wpli) / std_wpli for wpli in wpli_values]
    
    # Invert Z_Running_Time_s because smaller is better
    z_rt = [-z for z in z_rt]
    
    # Compute composite scores using alpha and gamma
    scores = [alpha * z_rt[i] + gamma * z_wpli[i] for i in range(len(subset))]
    
    # Normalize scores within the subset between 0 and 1
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        # If all scores are identical, assign normalized score of 1.0
        normalized_scores = [1.0 for _ in scores]
    else:
        normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
    
    # Update the profiling_data entries that match the current cache_calls category
    for i, entry in enumerate(subset):
        entry['Z_Running_Time_s'] = z_rt[i]
        entry['Z_WPLI'] = z_wpli[i]
        entry['Score'] = scores[i]
        entry['Normalized_Score'] = normalized_scores[i]
    
    return subset

