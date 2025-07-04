import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from ..ttns.ttndo import BRA_SUFFIX, KET_SUFFIX
from .binary import PHYS_PREFIX, VIRTUAL_PREFIX

def visualize_binary_ttns(ttns, title=None, curved_edges=True, save_path=None, layout_type="hierarchical", simplified_labels=False):
    """
    Create a visual representation of a Binary Tree Tensor Network State with improved layout.
    Uses intelligent node positioning to create clearer visualizations.
    
    Args:
        ttns: The binary tree tensor network state to visualize
        title: Optional title for the plot
        curved_edges: Whether to draw curved edges (helps avoid overlaps)
        save_path: Optional path to save the visualization
        layout_type: Type of layout to use ("hierarchical" or "radial")
        simplified_labels: If True, show only numeric parts of node names
    """

    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Organize nodes by level and type
    nodes_by_level = {}
    
    # Add all nodes to the graph and organize by level
    for node_id, node in ttns.nodes.items():
        # Determine node type
        is_physical = PHYS_PREFIX in node_id
        is_root = node_id == ttns.root_id
        
        # Add node with attributes
        G.add_node(node_id, is_physical=is_physical, is_root=is_root)
        
        # Determine level - try to extract from node ID
        if is_root:
            level = 0
        elif is_physical:
            level = float('inf')  # Will be adjusted later
        else:
            # Extract level from node ID (format: nodeX_Y)
            try:
                level = int(node_id.split('_')[0].replace(VIRTUAL_PREFIX, ''))
            except (ValueError, IndexError):
                level = -1
        
        # Add to level dictionary
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node_id)
    
    # Adjust physical nodes level
    if float('inf') in nodes_by_level:
        max_level = max([l for l in nodes_by_level.keys() if l != float('inf')], default=0)
        phys_nodes = nodes_by_level.pop(float('inf'))
        nodes_by_level[max_level + 1] = phys_nodes
    
    # Add edges (parent-child relationships)
    edges = []
    for node_id, node in ttns.nodes.items():
        for child_id in node.children:
            edges.append((node_id, child_id))
            G.add_edge(node_id, child_id)
    
    # Helper function to extract node index
    def extract_node_index(node_id):
        if PHYS_PREFIX in node_id:
            try:
                return int(''.join(filter(str.isdigit, node_id)))
            except ValueError:
                return 0
        else:
            try:
                parts = node_id.split('_')
                if len(parts) > 1:
                    return int(parts[1])
                return 0
            except (ValueError, IndexError):
                return 0
    
    # Create the layout based on type
    pos = {}  # Initialize pos dictionary here for all layout types
    levels = sorted(nodes_by_level.keys())  # Define levels for all layout types
    
    
    if layout_type == "radial":
        # Position nodes in concentric circles
        if not levels:
            return  # No nodes to draw
            
        max_level = max(levels)
        
        # Handle level 0 nodes (could be multiple)
        if 0 in nodes_by_level and nodes_by_level[0]:
            level0_nodes = sorted(nodes_by_level[0], key=extract_node_index)
            
            # Single node at level 0 - place at center
            if len(level0_nodes) == 1:
                pos[level0_nodes[0]] = (0.5, 0.5)
            # Multiple nodes at level 0 - arrange in a small circle
            else:
                small_radius = 0.03  # Small radius for level 0 nodes
                for i, node_id in enumerate(level0_nodes):
                    angle = 2 * np.pi * i / len(level0_nodes)
                    x = 0.5 + small_radius * np.cos(angle)
                    y = 0.5 + small_radius * np.sin(angle)
                    pos[node_id] = (x, y)
        
        # Position other nodes in concentric circles with even spacing
        for level_idx, level in enumerate(levels):
            if level == 0:
                continue  # Level 0 already positioned
                
            nodes = nodes_by_level[level]
            if not nodes:
                continue
                
            # Use an evenly distributed radius formula
            # This ensures equal spacing between levels (linear scaling)
            radius = 0.4 * level / max_level
            
            # Sort nodes for consistent positioning
            sorted_nodes = sorted(nodes, key=extract_node_index)
            
            # Calculate angles for evenly distributing the nodes
            total_nodes = len(sorted_nodes)
            
            for i, node_id in enumerate(sorted_nodes):
                angle = 2 * np.pi * i / total_nodes
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                pos[node_id] = (x, y)
    else:  # Default hierarchical layout
        # Find max level for scaling
        max_level = max(levels) if levels else 0
        
        # Get the physical nodes directly
        physical_nodes = [n for n, attr in G.nodes(data=True) if attr.get('is_physical', False)]
        phys_nodes = sorted(physical_nodes, key=extract_node_index)
        num_phys = len(phys_nodes)
        
        if num_phys > 0:
            for i, node_id in enumerate(phys_nodes):
                # Even spacing along the bottom
                x_pos = (i + 0.5) / (num_phys + 0.5)
                y_pos = 0.0  # Bottom of the plot
                pos[node_id] = (x_pos, y_pos)
        
        # Position virtual nodes level by level - lower level numbers (0,1) at top
        for level in sorted(levels):
            # Skip positioning physical nodes which we've already positioned
            level_nodes = [n for n in nodes_by_level[level] if not G.nodes[n].get('is_physical', False)]
            if not level_nodes:
                continue
                
            level_nodes = sorted(level_nodes, key=extract_node_index)
            
            # Set y-position for this level - invert so level 0 at top
            y_pos = 1.0 - (level / (max_level + 1.0))
            
            # Position nodes at this level
            num_nodes = len(level_nodes)
            for i, node_id in enumerate(level_nodes):
                # Even spacing along this level
                x_pos = (i + 0.5) / (num_nodes + 0.5)
                pos[node_id] = (x_pos, y_pos)

    # Fallback for unpositioned nodes
    for node_id in G.nodes():
        if node_id not in pos:
            pos[node_id] = (np.random.rand() * 0.2 + 0.4, np.random.rand() * 0.2 + 0.4)
    
    # Determine plot limits based on calculated positions (for radial layout)
    max_coord = 0.5 # Start with center
    if pos and layout_type == "radial":
        all_coords = np.array(list(pos.values()))
        max_coord = max(np.max(np.abs(all_coords[:,0] - 0.5)), np.max(np.abs(all_coords[:,1] - 0.5)))
    plot_radius = max(max_coord, 0.1) * 1.1 # Add 10% buffer, ensure minimum size
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Extract the different node types for coloring
    root_nodes = [n for n, attr in G.nodes(data=True) if attr.get('is_root', False)]
    phys_nodes = [n for n, attr in G.nodes(data=True) if attr.get('is_physical', False)]
    virt_nodes = [n for n in G.nodes() if n not in phys_nodes and n not in root_nodes]
    
    # Determine curved_edges parameter based on layout_type
    if layout_type == "radial":
        curved_edges = True  # Always use curved edges for radial layout
        curve_param = 0.15   # Stronger curve for radial layout
    else:
        curve_param = 0.1    # Default curve for hierarchical layout
    
    # Draw edges with curved connections for better separation if requested
    if curved_edges:
        nx.draw_networkx_edges(G, pos, 
                              connectionstyle=f'arc3,rad={curve_param}',
                              arrows=True,
                              arrowstyle='-',  # No arrow heads
                              alpha=0.7,
                              width=1.0 if layout_type == "hierarchical" else 0.8)  # Thinner for radial
    else:
        nx.draw_networkx_edges(G, pos, arrows=False, alpha=0.7, width=1.0)
    
    # Draw nodes with appropriate colors
    node_size = 400 if layout_type == "hierarchical" else 300  # Smaller nodes for radial layout
    nx.draw_networkx_nodes(G, pos, nodelist=root_nodes, node_color='#FFD700',  # Gold
                          node_size=node_size, label='Root')
    nx.draw_networkx_nodes(G, pos, nodelist=phys_nodes, node_color='#FF9999',  # Light red
                          node_size=node_size - 100, label='Physical')
    nx.draw_networkx_nodes(G, pos, nodelist=virt_nodes, node_color='#4b5eea',  # Light blue
                          node_size=node_size - 100, label='Virtual')
    
    # Generate simplified labels if requested
    if simplified_labels:
        labels = {}
        for node_id in G.nodes():
            if PHYS_PREFIX in node_id:
                # Extract only the numeric part for physical nodes
                labels[node_id] = ''.join(filter(str.isdigit, node_id))
            else:
                # Extract X_Y from nodeX_Y format
                try:
                    base_parts = node_id.split('_')
                    if len(base_parts) >= 2:
                        level = base_parts[0].replace(VIRTUAL_PREFIX, '')
                        idx = base_parts[1]
                        labels[node_id] = f"{level}_{idx}"
                    else:
                        labels[node_id] = node_id  # Fallback to original
                except:
                    labels[node_id] = node_id  # Fallback to original
    else:
        labels = None  # Use default node labels
    
    # Draw labels with smaller font for readability
    nx.draw_networkx_labels(G, pos, font_size=8, labels=labels)
    
    # Add legend
    plt.legend()
    
    if layout_type == "radial":
        # Set explicit plot limits for radial layout
        plt.xlim(0.5 - plot_radius, 0.5 + plot_radius)
        plt.ylim(0.5 - plot_radius, 0.5 + plot_radius)
        plt.gca().set_aspect('equal', adjustable='box') # Ensure circle isn't distorted
    
    plt.axis('off')
    
    layout_name = "Hierarchical" if layout_type == "hierarchical" else "Radial"
    if title:
        plt.title(title, pad=20)
    else:
        plt.title(f"Binary TTNS Structure ({layout_name} Layout)", pad=20)
    
    plt.tight_layout()

    
    plt.show()
def visualize_binary_ttndo(ttndo, title=None, save_path=None, layout_type="radial", simplified_labels=False):
    """
    Create a visual representation of a Binary Tree Tensor Network Density Operator.
    
    Args:
        ttndo: The binary tree tensor network density operator to visualize
        title: Optional title for the plot
        save_path: Optional path to save the visualization
        layout_type: Type of layout to use ("radial" or "hierarchical")
        simplified_labels: If True, show only numeric parts of node names
    """
    # Import required libraries
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Organize nodes by level
    nodes_by_level = {}
    
    # Track bra-ket pairs for potential later use
    bra_ket_pairs = []
    bra_nodes = set()
    ket_nodes = set()
    
    # Add all nodes to the graph and organize by level
    for node_id, node in ttndo.nodes.items():
        # Determine if this is a physical node (by convention)
        is_physical = PHYS_PREFIX in node_id 
        
        # Determine if this is a bra or ket node
        is_bra = BRA_SUFFIX in node_id
        is_ket = KET_SUFFIX in node_id
        is_root = node_id == ttndo.root_id # Check if root
        
        # Add node with attributes
        G.add_node(node_id, is_physical=is_physical, is_bra=is_bra, is_ket=is_ket, is_root=is_root)
        
        # Track node types
        if is_bra:
            bra_nodes.add(node_id)
        elif is_ket:
            ket_nodes.add(node_id)
        
        # Determine level from node ID
        if is_root:
            level = 0
        elif is_physical:
            # Place physical nodes at the bottom level
            level = float('inf')  # Will be adjusted later
        else:
            # For virtual nodes, extract level from ID (format: "nodeX_Y_bra" or "nodeX_Y_ket")
            try:
                # Remove bra/ket suffix for level extraction
                base_id = node_id.replace(BRA_SUFFIX, "").replace(KET_SUFFIX, "")
                level = int(base_id.split('_')[0].replace(VIRTUAL_PREFIX, ''))
            except (ValueError, IndexError):
                # If can't determine level, use a default
                level = -1
        
        # Add to level dictionary
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node_id)
        
        # Find and store bra-ket pairs
        if is_bra:
            ket_id = node_id.replace(BRA_SUFFIX, KET_SUFFIX)
            if ket_id in ttndo.nodes:
                bra_ket_pairs.append((node_id, ket_id))
    
    # If we have physical nodes, place them one level below the deepest virtual level
    if float('inf') in nodes_by_level:
        max_virtual_level = max([l for l in nodes_by_level.keys() if l != float('inf')], default=0)
        phys_nodes = nodes_by_level.pop(float('inf'))
        nodes_by_level[max_virtual_level + 1] = phys_nodes
    
    # Add regular edges (parent-child relationships)
    normal_edges = []
    for node_id, node in ttndo.nodes.items():
        # Get the list of children 
        for child_id in node.children:
            # Add edge from parent to child
            G.add_edge(node_id, child_id)
            normal_edges.append((node_id, child_id))
    
    # Function to extract node indices for sorting
    def extract_node_index(node_id):
        # Remove bra/ket suffix for comparison
        base_id = node_id.replace(BRA_SUFFIX, "").replace(KET_SUFFIX, "")
        
        if PHYS_PREFIX in base_id:
            # Extract numeric index from physical node (site5_bra -> 5)
            try:
                return int(''.join(filter(str.isdigit, base_id)))
            except ValueError:
                return 0
        else:
            # For virtual nodes (nodeX_Y_bra), extract Y
            try:
                return int(base_id.split('_')[1])
            except (ValueError, IndexError):
                return 0
    
    # Choose layout based on layout_type parameter
    pos = {}
    levels = sorted(nodes_by_level.keys())
    max_level = max(levels) if levels else 0
    
    if layout_type == "radial":
        # Position nodes in concentric circles
        for level_idx, level in enumerate(levels):
            nodes = nodes_by_level.get(level, [])
            if not nodes:
                continue
            
            # For level 0, arrange bra-ket pairs on opposite sides
            if level == 0:
                # Find bra-ket pairs at level 0
                level0_pairs = []
                level0_singles = []
                
                # Group nodes into pairs
                processed = set()
                for node_id in nodes:
                    if node_id in processed:
                        continue
                    
                    if BRA_SUFFIX in node_id:
                        ket_id = node_id.replace(BRA_SUFFIX, KET_SUFFIX)
                        if ket_id in nodes:
                            level0_pairs.append((node_id, ket_id))
                            processed.add(node_id)
                            processed.add(ket_id)
                        else:
                            level0_singles.append(node_id)
                            processed.add(node_id)
                    elif KET_SUFFIX in node_id:
                        bra_id = node_id.replace(KET_SUFFIX, BRA_SUFFIX)
                        if bra_id in nodes:
                            level0_pairs.append((bra_id, node_id))
                            processed.add(bra_id)
                            processed.add(node_id)
                        else:
                            level0_singles.append(node_id)
                            processed.add(node_id)
                    else:
                        level0_singles.append(node_id)
                        processed.add(node_id)
                
                # Position pairs on opposite sides
                radius = 0.06  # Small radius for level 0 nodes, matching intertwined TTNDO
                pair_count = len(level0_pairs)
                single_count = len(level0_singles)
                
                # Handle pairs first - position on opposite sides
                for i, (bra_id, ket_id) in enumerate(level0_pairs):
                    angle = i * (np.pi / pair_count) if pair_count > 1 else 0
                    
                    # Bra node at angle
                    x_bra = 0.5 + radius * np.cos(angle)
                    y_bra = 0.5 + radius * np.sin(angle)
                    pos[bra_id] = (x_bra, y_bra)
                    
                    # Ket node at opposite side
                    x_ket = 0.5 + radius * np.cos(angle + np.pi)
                    y_ket = 0.5 + radius * np.sin(angle + np.pi)
                    pos[ket_id] = (x_ket, y_ket)
                
                # Handle singles - position in between pairs if needed
                if single_count > 0:
                    offset = np.pi / 2
                    for i, node_id in enumerate(level0_singles):
                        angle = offset + i * (2 * np.pi / single_count)
                        x = 0.5 + radius * np.cos(angle)
                        y = 0.5 + radius * np.sin(angle)
                        pos[node_id] = (x, y)
                
                continue
            
            # Use specific radii for each level, similar to visualize_binary_ttndo
            radii = {
                0: 0.00,    # Level 0 at center (handled above)
                1: 0.12,    # Level 1 close to center 
                2: 0.20,    # Level 2
                3: 0.28,    # Level 3
                4: 0.35,    # Level 4
                5: 0.40,    # Level 5
                6: 0.50,    # Level 6
            }
            
            # Get radius for this level (use dict values, or calculate for higher levels)
            radius = radii.get(level, 0.1 * level)
            
            sorted_nodes = sorted(nodes, key=extract_node_index)
            total_nodes = len(sorted_nodes)
    
            if total_nodes > 0:
                angle_step = 2 * np.pi / total_nodes
                for i, node_id in enumerate(sorted_nodes):
                    angle = i * angle_step
                    x = 0.5 + radius * np.cos(angle)
                    y = 0.5 + radius * np.sin(angle)
                    pos[node_id] = (x, y)
    
    else:  # hierarchical layout - use the bottom-up positioning from our current function
        # Position physical nodes first at the bottom
        physical_level = max_level # Physical nodes are at the highest level index
        if physical_level in nodes_by_level:
            phys_nodes = nodes_by_level[physical_level]
            # Group physical nodes by base ID for paired positioning
            phys_grouped = {}
            for node_id in phys_nodes:
                base_id = node_id.replace(BRA_SUFFIX, "").replace(KET_SUFFIX, "")
                if base_id not in phys_grouped:
                    phys_grouped[base_id] = []
                phys_grouped[base_id].append(node_id)
                
            sorted_phys_bases = sorted(phys_grouped.keys(), key=extract_node_index)
            num_phys_bases = len(sorted_phys_bases)
            x_step = 1.0 / (num_phys_bases + 1)
            bra_ket_offset = 0.03 # Small vertical offset
    
            for i, base_id in enumerate(sorted_phys_bases):
                x_pos = (i + 1) * x_step
                y_pos = 0 # Bottom of plot
                nodes_in_group = phys_grouped[base_id]
                for node_id in nodes_in_group:
                    if BRA_SUFFIX in node_id:
                        pos[node_id] = (x_pos, y_pos - bra_ket_offset)
                    elif KET_SUFFIX in node_id:
                        pos[node_id] = (x_pos, y_pos + bra_ket_offset)
                    else: # Should not happen for physical in TTNDO but handle anyway
                        pos[node_id] = (x_pos, y_pos)
        
        # Position internal nodes level by level, bottom-up
        for level in reversed(levels):
            if level == physical_level: # Skip physical level, already done
                continue
            
            nodes = nodes_by_level.get(level, [])
            if not nodes:
                continue
                
            # Group and sort nodes by their base name (without bra/ket suffix)
            grouped_nodes = {}
            for node_id in nodes:
                base_id = node_id.replace(BRA_SUFFIX, "").replace(KET_SUFFIX, "")
                if base_id not in grouped_nodes:
                    grouped_nodes[base_id] = []
                grouped_nodes[base_id].append(node_id)
            
            # Sort the groups by their numerical index
            sorted_groups = sorted(grouped_nodes.keys(), key=extract_node_index)
            
            # Position each node group (bra/ket pair or single node like root)
            for base_id in sorted_groups:
                nodes_in_group = grouped_nodes[base_id]
                # Try positioning based on children first
                child_positions = []
                primary_node = nodes_in_group[0] # Use first node in group to find children
                node_obj = ttndo.nodes[primary_node]
                children = node_obj.children
                
                if children and all(child in pos for child in children):
                    child_positions = [pos[child][0] for child in children]
                    x_pos = sum(child_positions) / len(child_positions)
                else:
                    # Fallback: position based on index within the level
                    num_bases_level = len(sorted_groups)
                    base_idx = sorted_groups.index(base_id)
                    x_pos = (base_idx + 0.5) / (num_bases_level + 1)
    
                # Calculate y position based on level (higher levels = higher y)
                y_pos = (max_level - level) / max_level if max_level > 0 else 0.5
                
                # Assign position to nodes in the group with vertical offset
                bra_ket_offset = 0.03 # Consistent vertical offset
                for node_id in nodes_in_group:
                    if BRA_SUFFIX in node_id:
                        pos[node_id] = (x_pos, y_pos - bra_ket_offset)
                    elif KET_SUFFIX in node_id:
                        pos[node_id] = (x_pos, y_pos + bra_ket_offset)
                    else: # Root node
                        pos[node_id] = (x_pos, y_pos)
    
        # Adjust positions to prevent overlaps (like in visualize_binary_ttns)
        for level in levels:
            nodes_at_level = nodes_by_level.get(level, [])
            if len(nodes_at_level) <= 1:
                continue
    
            # Sort nodes horizontally based on their current tentative positions
            sorted_level_nodes = sorted(nodes_at_level, key=lambda n: pos.get(n, (0,0))[0])
    
            min_gap = 0.08 # Minimum horizontal gap
            for i in range(1, len(sorted_level_nodes)):
                curr_node = sorted_level_nodes[i]
                prev_node = sorted_level_nodes[i-1]
                if curr_node not in pos or prev_node not in pos: continue # Skip if node wasn't positioned
    
                curr_x, curr_y = pos[curr_node]
                prev_x, prev_y = pos[prev_node]
    
                if curr_x - prev_x < min_gap:
                    pos[curr_node] = (prev_x + min_gap, curr_y)
    
        # Center each level horizontally (like in visualize_binary_ttns)
        for level in levels:
            nodes_at_level = nodes_by_level.get(level, [])
            if not nodes_at_level:
                continue
                
            valid_nodes = [n for n in nodes_at_level if n in pos]
            if not valid_nodes:
                continue
                
            x_positions = [pos[node][0] for node in valid_nodes]
            if not x_positions: continue
            
            min_x = min(x_positions)
            max_x = max(x_positions)
            level_width = max_x - min_x
            current_center = min_x + level_width / 2
            target_center = 0.5
            shift = target_center - current_center
    
            for node_id in valid_nodes:
                x, y = pos[node_id]
                pos[node_id] = (x + shift, y)

    # Fallback for any unpositioned nodes
    for node_id in G.nodes():
        if node_id not in pos:
            print(f"Warning: Node {node_id} in binary TTNDO was not positioned. Using default.")
            pos[node_id] = (np.random.rand() * 0.2 + 0.4, np.random.rand() * 0.2 + 0.4)
    
    # Determine plot limits for radial layout
    max_coord = 0.5 # Start with center
    if layout_type == "radial" and pos:
        all_coords = np.array(list(pos.values()))
        max_coord = max(np.max(np.abs(all_coords[:,0] - 0.5)), np.max(np.abs(all_coords[:,1] - 0.5)))
    plot_radius = max(max_coord, 0.1) * 1.1 # Add 10% buffer, ensure minimum size
    
    # Set up the figure
    plt.figure(figsize=(12, 10))
    
    # Separate regular edges and bra-ket edges for visualization
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('is_bra_ket', False)]
    
    # Draw all edges with the same style for consistency
    nx.draw_networkx_edges(G, pos, 
                          edgelist=normal_edges,
                          arrows=True, 
                          arrowstyle='-', # No arrowheads
                          connectionstyle='arc3,rad=0.15', # Consistent curve for all edges
                          alpha=0.7,
                          width=0.8) # Thinner edges
        
    # Draw nodes with different colors
    root_nodes = [n for n, attr in G.nodes(data=True) if attr.get('is_root', False)]
    phys_ket_nodes = [n for n, attr in G.nodes(data=True) 
                     if attr.get('is_physical', False) and attr.get('is_ket', False)]
    phys_bra_nodes = [n for n, attr in G.nodes(data=True) 
                     if attr.get('is_physical', False) and attr.get('is_bra', False)]
    virt_ket_nodes = [n for n, attr in G.nodes(data=True) 
                     if not attr.get('is_physical', False) and attr.get('is_ket', False)]
    virt_bra_nodes = [n for n, attr in G.nodes(data=True) 
                     if not attr.get('is_physical', False) and attr.get('is_bra', False)]
    other_nodes = [n for n in G.nodes() 
                  if n not in root_nodes + phys_ket_nodes + phys_bra_nodes + 
                  virt_ket_nodes + virt_bra_nodes]
    
    # Determine node sizes based on layout
    root_size = 300
    reg_size = 200 if layout_type == "radial" else 300
    
    # Draw nodes with different colors
    nx.draw_networkx_nodes(G, pos, nodelist=root_nodes, node_color='#FFD700',  # Gold for root
                          node_size=root_size, label='Root')
    nx.draw_networkx_nodes(G, pos, nodelist=phys_ket_nodes, node_color='#FF9999',  # Light red
                          node_size=reg_size, label='Physical Ket')
    nx.draw_networkx_nodes(G, pos, nodelist=phys_bra_nodes, node_color='#FF99FF',  # Light pink
                          node_size=reg_size, label='Physical Bra')
    nx.draw_networkx_nodes(G, pos, nodelist=virt_ket_nodes, node_color='#4b5eea',  # Light blue
                          node_size=reg_size, label='Virtual Ket')
    nx.draw_networkx_nodes(G, pos, nodelist=virt_bra_nodes, node_color='#99CCFF',  # Very light blue
                          node_size=reg_size, label='Virtual Bra')
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color='#4b5eea',  # Blue for virtual nodes
                          node_size=reg_size, label='Virtual')
    
    # Generate simplified labels if requested
    if simplified_labels:
        labels = {}
        for node_id in G.nodes():
            base_id = node_id.replace(BRA_SUFFIX, "").replace(KET_SUFFIX, "")
            
            if PHYS_PREFIX in base_id:
                # Extract only the numeric part for physical nodes
                num_part = ''.join(filter(str.isdigit, base_id))
                labels[node_id] = num_part
            else:
                # Extract X_Y from nodeX_Y format
                try:
                    base_parts = base_id.split('_')
                    if len(base_parts) >= 2:
                        level = base_parts[0].replace(VIRTUAL_PREFIX, '')
                        idx = base_parts[1]
                        labels[node_id] = f"{level}_{idx}"
                    else:
                        labels[node_id] = node_id  # Fallback to original
                except:
                    labels[node_id] = node_id  # Fallback to original
    else:
        labels = None  # Use default node labels
    
    # Draw labels with smaller font for readability
    nx.draw_networkx_labels(G, pos, font_size=8, labels=labels)
    
    plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=3)
    
    # For radial layout, set explicit plot limits
    if layout_type == "radial":
        plt.xlim(0.5 - plot_radius, 0.5 + plot_radius)
        plt.ylim(0.5 - plot_radius, 0.5 + plot_radius)
        plt.gca().set_aspect('equal', adjustable='box') # Ensure circle isn't distorted
    
    plt.axis('off')
    
    layout_name = "Hierarchical" if layout_type == "hierarchical" else "Radial"
    if title:
        plt.title(title, pad=20)
    else:
        plt.title(f"Binary TTNDO Structure ({layout_name} Layout)", pad=20)
        
    plt.tight_layout()
    
    
    plt.show()
def visualize_symmetric_ttndo(ttndo, title=None, save_path=None, simplified_labels=False):
    """
    Create a visual representation of a Symmetric Tree Tensor Network Density Operator.
    Uses a specialized radial layout with nodes arranged in concentric circles based on the implementation in Sample.ipynb.
    
    Args:
        ttndo: The symmetric tree tensor network density operator to visualize
        title: Optional title for the plot
        save_path: Optional path to save the visualization
        simplified_labels: If True, show only numeric parts of node names
    """
    # Import required libraries
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Organize nodes by level and type
    nodes_by_level = {}
    bra_nodes = set()
    ket_nodes = set()
    
    # Add all nodes to the graph and organize by level
    for node_id, node in ttndo.nodes.items():
        # Determine node type
        is_physical = PHYS_PREFIX in node_id
        is_bra = BRA_SUFFIX in node_id
        is_ket = KET_SUFFIX in node_id
        is_root = node_id == ttndo.root_id
        
        # Add node with attributes
        G.add_node(node_id, is_physical=is_physical, is_bra=is_bra, 
                  is_ket=is_ket, is_root=is_root)
        
        # Track bra/ket nodes
        if is_bra:
            bra_nodes.add(node_id)
        elif is_ket:
            ket_nodes.add(node_id)
        
        # Determine level based on node ID structure
        if is_root:
            level = 0
        elif is_physical:
            level = float('inf')  # Will be adjusted later
        else:
            # Extract level from node ID (format: nodeX_Y_[bra/ket])
            try:
                base_id = node_id.replace(BRA_SUFFIX, "").replace(KET_SUFFIX, "")
                # Add 1 to level to account for root at level 0
                level = int(base_id.split('_')[0].replace(VIRTUAL_PREFIX, '')) + 1  
            except (ValueError, IndexError):
                level = -1 # Default level if ID format is unexpected
        
        # Add to level dictionary
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node_id)
    
    # Adjust physical nodes level
    if float('inf') in nodes_by_level:
        max_virtual_level = max([l for l in nodes_by_level.keys() if l != float('inf')], default=0)
        phys_nodes = nodes_by_level.pop(float('inf'))
        nodes_by_level[max_virtual_level + 1] = phys_nodes
    
    # Add edges (parent-child relationships)
    edges = []
    for node_id, node in ttndo.nodes.items():
        for child_id in node.children:
            edges.append((node_id, child_id))
            G.add_edge(node_id, child_id)

    # Helper function to extract node index
    def extract_node_index(node_id):
        base_id = node_id.replace(BRA_SUFFIX, "").replace(KET_SUFFIX, "")
        if PHYS_PREFIX in base_id:
            try:
                return int(''.join(filter(str.isdigit, base_id)))
            except ValueError:
                return 0
        else:
            try:
                # Extract the second part of the ID (Y in nodeX_Y)
                return int(base_id.split('_')[1])
            except (ValueError, IndexError):
                return 0
    
    # Create the radial layout based on Sample.ipynb
    pos = {}
    levels = sorted(nodes_by_level.keys())
    
    # Position nodes in concentric circles
    for level_idx, level in enumerate(levels):
        if level not in nodes_by_level or not nodes_by_level[level]:
            continue
        nodes = nodes_by_level[level]
        
        # Calculate radius - use a simple scaling that increases with level
        radius = 0.4 * level_idx # Simple linear scaling with level index
        
        # Separate nodes by type for this level
        bra_level_nodes = [n for n in nodes if n in bra_nodes]
        ket_level_nodes = [n for n in nodes if n in ket_nodes]
        other_level_nodes = [n for n in nodes if n not in bra_nodes and n not in ket_nodes]
        
        # Sort each category by their base index
        bra_level_nodes.sort(key=extract_node_index)
        ket_level_nodes.sort(key=extract_node_index)
        other_level_nodes.sort(key=extract_node_index)
        
        # Position root at center
        if level == 0:
            if other_level_nodes:  # Root node is neither bra nor ket
                pos[other_level_nodes[0]] = (0.5, 0.5)  # Center
            elif nodes:  # If no 'other' nodes, take the first one
                pos[nodes[0]] = (0.5, 0.5)
            continue
            
        # For all other levels - distribute around the full circle
        all_nodes = []
        
        # First add all bra nodes, then all ket nodes to ensure they're grouped
        all_nodes.extend(bra_level_nodes)
        all_nodes.extend(ket_level_nodes)
        all_nodes.extend(other_level_nodes)
        
        # Now position all nodes evenly around the circle
        node_count = len(all_nodes)
        if node_count > 0:
            angle_step = 2 * np.pi / node_count
            for i, node_id in enumerate(all_nodes):
                angle = i * angle_step
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                pos[node_id] = (x, y)

    # Fallback for any unpositioned nodes
    for node_id in G.nodes():
        if node_id not in pos:
            print(f"Warning: Node {node_id} in symmetric TTNDO was not positioned. Using default.")
            pos[node_id] = (np.random.rand() * 0.2 + 0.4, np.random.rand() * 0.2 + 0.4)
    
    # Determine plot limits based on calculated positions
    max_coord = 0.5 # Start with center
    if pos:
        all_coords = np.array(list(pos.values()))
        max_coord = max(np.max(np.abs(all_coords[:,0] - 0.5)), np.max(np.abs(all_coords[:,1] - 0.5)))
    plot_radius = max(max_coord, 0.1) * 1.1 # Add 10% buffer, ensure minimum size
    
    # Set up the figure
    plt.figure(figsize=(12, 12))
    
    # Extract node types for coloring
    root_nodes = [n for n, attr in G.nodes(data=True) if attr.get('is_root', False)]
    phys_bra_nodes = [n for n, attr in G.nodes(data=True) 
                     if attr.get('is_physical', False) and attr.get('is_bra', False)]
    phys_ket_nodes = [n for n, attr in G.nodes(data=True) 
                     if attr.get('is_physical', False) and attr.get('is_ket', False)]
    virt_bra_nodes = [n for n, attr in G.nodes(data=True) 
                     if not attr.get('is_physical', False) and attr.get('is_bra', False)]
    virt_ket_nodes = [n for n, attr in G.nodes(data=True) 
                     if not attr.get('is_physical', False) and attr.get('is_ket', False)]
    other_nodes = [n for n in G.nodes() 
                  if n not in root_nodes + phys_bra_nodes + phys_ket_nodes + 
                  virt_bra_nodes + virt_ket_nodes]
    
    # Separate normal edges for drawing (no bra-ket edges)
    normal_edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if not d.get('is_bra_ket', False)]
    
    # Draw normal edges (solid lines)
    nx.draw_networkx_edges(G, pos, 
                          edgelist=normal_edges_to_draw,
                          arrows=True,
                          arrowstyle='-', # No arrow head for TTNDO visualization
                          connectionstyle='arc3,rad=0.1', # Slight curve
                          alpha=0.7,
                          width=0.8)
    
    # Draw nodes with appropriate colors
    nx.draw_networkx_nodes(G, pos, nodelist=root_nodes, node_color='#FFD700',  # Gold
                          node_size=300, label='Root')
    nx.draw_networkx_nodes(G, pos, nodelist=phys_bra_nodes, node_color='#FF99FF',  # Light Pink
                          node_size=200, label='Physical Bra')
    nx.draw_networkx_nodes(G, pos, nodelist=phys_ket_nodes, node_color='#FF9999',  # Light Red
                          node_size=200, label='Physical Ket')
    nx.draw_networkx_nodes(G, pos, nodelist=virt_bra_nodes, node_color='#99CCFF',  # Very Light Blue
                          node_size=250, label='Virtual Bra')
    nx.draw_networkx_nodes(G, pos, nodelist=virt_ket_nodes, node_color='#4b5eea',  # Light Blue
                          node_size=250, label='Virtual Ket')
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color='#4b5eea',  # Blue for virtual nodes
                          node_size=200, label='Virtual')
    
    # Generate simplified labels if requested
    if simplified_labels:
        labels = {}
        for node_id in G.nodes():
            base_id = node_id.replace(BRA_SUFFIX, "").replace(KET_SUFFIX, "")
            
            if PHYS_PREFIX in base_id :
                # Extract only the numeric part for physical nodes
                num_part = ''.join(filter(str.isdigit, base_id))
                labels[node_id] = num_part
            else:
                # Extract X_Y from nodeX_Y format
                try:
                    base_parts = base_id.split('_')
                    if len(base_parts) >= 2:
                        level = base_parts[0].replace(VIRTUAL_PREFIX, '')
                        idx = base_parts[1]
                        labels[node_id] = f"{level}_{idx}"
                    else:
                        labels[node_id] = node_id  # Fallback to original
                except:
                    labels[node_id] = node_id  # Fallback to original
    else:
        labels = None  # Use default node labels
    
    # Draw labels with smaller font for readability
    nx.draw_networkx_labels(G, pos, font_size=8, labels=labels)
    
    # Add legend with multiple columns for better layout
    plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=3)
    
    # Set explicit plot limits
    plt.xlim(0.5 - plot_radius, 0.5 + plot_radius)
    plt.ylim(0.5 - plot_radius, 0.5 + plot_radius)
    plt.gca().set_aspect('equal', adjustable='box') # Ensure circle isn't distorted
    
    plt.axis('off')
    
    if title:
        plt.title(title, pad=20)
    else:
        plt.title("Symmetric TTNDO Structure (Radial Layout)", pad=20)
    
    plt.tight_layout()
    
    
    plt.show() 
