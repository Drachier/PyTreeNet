from typing import Tuple, Any, Dict, List
from enum import Enum

import copy
import math
import random
import numpy as np
from copy import deepcopy
from ...operators.common_operators import bosonic_operators , pauli_matrices
from ...operators import Hamiltonian, TensorProduct
from ...random.random_matrices import crandn
from ...core.leg_specification import LegSpecification
from ...util.tensor_splitting import SVDParameters , ContractionMode , SplitMode
from ...ttns import TreeTensorNetworkState
from ...ttno.ttno_class import TTNO


def get_neighbors_with_distance_HV(Lx, Ly, distance):
  current_sites = []  # List to store current sites
  neighbor_sites = []  # List to store neighbor sites

  for i in range(Lx):
      for j in range(Ly):
          current_site = (i, j)
          # Check possible neighbor offsets
          for di in range(-distance, distance + 1):
              for dj in range(-distance, distance + 1):
                  # Only consider neighbors that are exactly at the specified distance
                  if math.sqrt(di**2 + dj**2) == distance:
                      neighbor_site = (i + di, j + dj)
                      # Check if the neighbor is within bounds
                      if 0 <= neighbor_site[0] < Lx and 0 <= neighbor_site[1] < Ly:
                          # Ensure we only add each unique connection once
                          if current_site < neighbor_site:  
                              current_sites.append(current_site)
                              neighbor_sites.append(neighbor_site)

  return current_sites, neighbor_sites  # Return as a list with two elements

def get_neighbors_with_distance_HDV(Lx, Ly, distance):
  current_sites = []  # List to store current sites
  neighbor_sites = []  # List to store neighbor sites

  # Define the allowed distances
  if distance == 1:
      allowed_distances = [1, math.sqrt(2)]  # Both orthogonal and diagonal
  elif distance == 2:
      allowed_distances = [2, math.sqrt(8)]  # Both orthogonal and diagonal (2√2)
  else:
      allowed_distances = [distance]

  for i in range(Lx):
      for j in range(Ly):
          current_site = (i, j)
          # Check possible neighbor offsets
          for di in range(-distance, distance + 1):
              for dj in range(-distance, distance + 1):
                  # Calculate the exact Euclidean distance
                  exact_distance = math.sqrt(di**2 + dj**2)
                  
                  # Check if the distance matches any of the allowed distances
                  if any(abs(exact_distance - d) < 1e-10 for d in allowed_distances):
                      neighbor_site = (i + di, j + dj)
                      # Check if the neighbor is within bounds
                      if 0 <= neighbor_site[0] < Lx and 0 <= neighbor_site[1] < Ly:
                          # Ensure we only add each unique connection once
                          if current_site < neighbor_site:
                              current_sites.append(current_site)
                              neighbor_sites.append(neighbor_site)

  return current_sites, neighbor_sites


# T3N <-> TTN CONVERSIONS
class T3NMode(Enum):
      QR = "QR"
      SVD = "SVD"

def ttn_to_t3n(ttn, 
               T3N_dict = None, 
               T3N_mode = T3NMode.QR, 
               T3N_contr_mode = ContractionMode.EQUAL): 
    nodes = deepcopy(ttn.nodes)
    ttn_copy = deepcopy(ttn)
    dict = {}
    for node_id in nodes:
        node = ttn_copy.nodes[node_id]
        if node.nneighbours() > 2:
            if T3N_dict is not None:
                neighbour_id = T3N_dict[node_id]
            else: 
                children = [child for child in node.children]
                neighbour_id = np.random.choice(children)
            main_legs, next_legs = build_leg_specs(node, neighbour_id)
            if T3N_mode == T3NMode.SVD:
                ttn_copy.split_node_svd(node_id, main_legs, next_legs,
                    u_identifier= "3_" + node_id,
                    v_identifier=node_id,
                    svd_params = SVDParameters(max_bond_dim = np.inf , rel_tol= -np.inf , total_tol = -np.inf),
                    contr_mode = T3N_contr_mode)
            elif T3N_mode == T3NMode.QR:
                ttn_copy.split_node_qr(node_id, main_legs, next_legs,
                                       q_identifier= "3_" + node_id,
                                       r_identifier= node_id)  
                 
            shape = ttn_copy.tensors["3_" + node_id].shape
            if isinstance(ttn_copy , TreeTensorNetworkState):
                T = ttn_copy.tensors["3_" + node_id].reshape(shape + (1,))
                ttn_copy.tensors["3_" + node_id] = T 
                ttn_copy.nodes["3_" + node_id].link_tensor(T)
            elif isinstance(ttn_copy , TTNO):    
                T = ttn_copy.tensors["3_" + node_id].reshape(shape + (1,1))
                ttn_copy.tensors["3_" + node_id] = T 
                ttn_copy.nodes["3_" + node_id].link_tensor(T)
            dict[node_id] = neighbour_id
    return ttn_copy , dict

def t3n_to_ttn(state, node_map):
    ttn = deepcopy(state)
    for node_id in node_map:
        if isinstance(ttn , TreeTensorNetworkState):
           T = ttn.tensors["3_"+node_id].reshape(ttn.tensors["3_"+node_id].shape[:-1]) 
        elif isinstance(ttn , TTNO):
           T = ttn.tensors["3_"+node_id].reshape(ttn.tensors["3_"+node_id].shape[:-2])   
        ttn.tensors["3_"+node_id] = T
        ttn.nodes["3_"+node_id].link_tensor(T)
        ttn.contract_nodes("3_"+node_id, node_id, node_id)
    return ttn

def build_leg_specs(node ,
                        min_neighbour_id: str) -> Tuple[LegSpecification,LegSpecification]:
    """
    Construct the leg specifications required for the qr decompositions during
     canonicalisation.

    Args:
        node (Node): The node which is to be split.
        min_neighbour_id (str): The identifier of the neighbour of the node
         which is closest to the orthogonality center.

    Returns:
        Tuple[LegSpecification,LegSpecification]: 
            The leg specifications for the legs of the Q-tensor, i.e. what
            remains as the node, and the R-tensor, i.e. what will be absorbed
            into the node defined by `min_neighbour_id`.
    """
    main_legs = LegSpecification(None, copy.copy(node.children), [])
    if node.is_child_of(min_neighbour_id):
        next_legs = LegSpecification(min_neighbour_id, [], node.open_legs)
    else:
        main_legs.parent_leg = node.parent
        main_legs.child_legs.remove(min_neighbour_id)
        next_legs = LegSpecification(None, [min_neighbour_id], node.open_legs)
    if node.is_root():
        main_legs.is_root = True
    return main_legs, next_legs


# INITIAL STATES
def get_checkerboard_pattern(Lx, Ly):
    black_sites = []
    white_sites = []

    for x in range(Lx):
        for y in range(Ly):
            current_site = f"({x},{y})"
            if (x + y) % 2 == 0:
                black_sites.append(current_site)
            else:
                white_sites.append(current_site)

    # Output the sites in each category
    return black_sites, white_sites

def get_random_half_sites(Lx, Ly): 
    total_sites = Lx * Ly

    # Generate a list of all sites
    all_sites = [f"({x},{y})" for x in range(Lx) for y in range(Ly)]

    # Shuffle and split into two halves
    random.shuffle(all_sites)
    half = total_sites // 2

    black_sites = all_sites[:half]
    white_sites = all_sites[half:]

    return black_sites, white_sites

def uniform_product_state(ttn,
                          local_state, 
                          bond_dim = 2): 
    product_state = deepcopy(ttn)
    for node_id in product_state.nodes.keys():
        n = product_state.tensors[node_id].ndim - 1
        tensor = local_state.reshape((1,) * n + (2,))
        T = np.pad(tensor, n*((bond_dim, bond_dim),) + ((0, 0),))
        product_state.tensors[node_id] = T
        product_state.nodes[node_id].link_tensor(T)  
    return product_state
    
def alternating_product_state(ttn, 
                              black_state,
                              white_state,
                              bond_dim = 2, 
                              pattern = "checkerboard",
                              seed = None,
                              lattice_dim = 2): 
    if lattice_dim == 2:
        Lx = int(np.sqrt(len(ttn.nodes.keys())))
        Ly = Lx
    elif lattice_dim == 1:
        Lx = 1
        Ly = len(ttn.nodes.keys())   
    product_state = deepcopy(ttn)
    if pattern == "checkerboard":
       black_sites , white_sites = get_checkerboard_pattern(Lx,Ly)
    elif pattern == "half_random":
       random.seed(seed)
       black_sites , white_sites = get_random_half_sites(Lx,Ly)
    for node in black_sites:
        black_id = "Site" + f"{node}"
        n = product_state.tensors[black_id].ndim - 1
        tensor = black_state.reshape((1,) * n + (2,))
        T = np.pad(tensor, n*((bond_dim, bond_dim),) + ((0, 0),))
        product_state.tensors[black_id] = T
        product_state.nodes[black_id].link_tensor(T)  
    for node in white_sites:
        white_id = "Site" + f"{node}"
        n = product_state.tensors[white_id].ndim - 1
        tensor = white_state.reshape((1,) * n + (2,))
        T = np.pad(tensor, n*((bond_dim, bond_dim),) + ((0, 0),))
        product_state.tensors[white_id] = T
        product_state.nodes[white_id].link_tensor(T)  

    return product_state


# HAMILTONIANS
def BoseHubbard_ham(t, U, m, Lx, Ly , d):
    creation_op, annihilation_op, number_op = bosonic_operators(dimension=d)
    
    conversion_dict = {
        "b^dagger": creation_op,
        "b": annihilation_op,
        "n": number_op,
        f"I{d}": np.eye(d)
    }
    
    conversion_dict.update({
        "-t * b^dagger": -t * creation_op,
        "-t * b": -t * annihilation_op,
        "U * n * (n - 1)": U * number_op @ (number_op - np.eye(d)),
        "m*n": - m * number_op,
    })
    
    terms = []
    
    # Hopping terms
    for x in range(Lx):
        for y in range(Ly):
            current_site = f"Site({x},{y})"
            
            # Horizontal connections
            if x < Lx - 1:
                right_neighbor = f"Site({x+1},{y})"
                terms.append(TensorProduct({current_site: "-t * b^dagger", right_neighbor: "b"}))
                terms.append(TensorProduct({current_site: "-t * b", right_neighbor: "b^dagger"}))
            
            # Vertical connections
            if y < Ly - 1:
                up_neighbor = f"Site({x},{y+1})"
                terms.append(TensorProduct({current_site: "-t * b^dagger", up_neighbor: "b"}))
                terms.append(TensorProduct({current_site: "-t * b", up_neighbor: "b^dagger"}))
    
    # On-site interaction terms
    for x in range(Lx):
        for y in range(Ly):
            current_site = f"Site({x},{y})"
            terms.append(TensorProduct({current_site: "U * n * (n - 1)"}))
            terms.append(TensorProduct({current_site: "m*n"}))
    
    return Hamiltonian(terms, conversion_dict)

def get_neighbors_periodic(x, y, Lx, Ly):
  neighbors = []
  
  # Right neighbor (with periodic boundary)
  right_x = (x + 1) % Lx
  neighbors.append((f"Site({right_x},{y})"))
  
  # Up neighbor (with periodic boundary)
  up_y = (y + 1) % Ly
  neighbors.append((f"Site({x},{up_y})"))

def Anisotropic_Heisenberg_ham(J_x, J_y, J_z, h_z, Lx, Ly , boundary_condition = None):
    # Get the Pauli matrices
    X, Y, Z = pauli_matrices()
    
    # Create a conversion dictionary for the operators
    conversion_dict = {
        "X": X,
        "J_x * X": J_x * X,
        "Y": Y,
        "J_y * Y": J_y * Y,
        "Z": Z,
        "J_z * Z": J_z * Z,
        "I2": np.eye(2),
        "h_z * Z": h_z * Z
    }
    
    terms = []
    
    if boundary_condition == "periodic":
        for x in range(Lx):
            for y in range(Ly):
                current_site = f"Site({x},{y})"
                neighbors = get_neighbors_periodic(x, y, Lx, Ly)
                
                for neighbor in neighbors:
                    terms.append(TensorProduct({current_site: "X", neighbor: "J_x * X"}))
                    terms.append(TensorProduct({current_site: "Y", neighbor: "J_y * Y"}))
                    terms.append(TensorProduct({current_site: "Z", neighbor: "J_z * Z"})) 
    else:
        # Hopping terms
        for x in range(Lx):
            for y in range(Ly):
                current_site = f"Site({x},{y})"  

                # Horizontal connections
                if x < Lx - 1:
                    right_neighbor = f"Site({x+1},{y})" 
                    terms.append(TensorProduct({current_site: "X", right_neighbor: "J_x * X"}))
                    terms.append(TensorProduct({current_site: "Y", right_neighbor: "J_y * Y"}))
                    terms.append(TensorProduct({current_site: "Z", right_neighbor: "J_z * Z"})) 

                # Vertical connections
                if y < Ly - 1:
                    up_neighbor = f"Site({x},{y+1})"
                    terms.append(TensorProduct({current_site: "X", right_neighbor: "J_x * X"}))
                    terms.append(TensorProduct({current_site: "Y", right_neighbor: "J_y * Y"}))
                    terms.append(TensorProduct({current_site: "Z", right_neighbor: "J_z * Z"}))                     


    # On-site magnetic field terms
    for x in range(Lx):
        for y in range(Ly):
            current_site = f"Site({x},{y})"
            terms.append(TensorProduct({current_site: "h_z * Z"}))
    
    return Hamiltonian(terms, conversion_dict)


# OPERATORS (ptn.Hamiltonian)
def spatial_correlation_function(Lx, Ly, dist, dim, mode = "HV"):
    # <b^†_i * b_j> - not normalized
    if mode == "HV":
        current_sites, neighbor_sites = get_neighbors_with_distance_HV(Lx, Ly, dist)
    elif mode == "HDV":
        current_sites, neighbor_sites = get_neighbors_with_distance_HDV(Lx, Ly, dist)

    # Define operators
    creation_op, annihilation_op, number_op = bosonic_operators(dim)
    conversion_dict = {
        "b^dagger": creation_op / len(current_sites),
        "b": annihilation_op,
        f"I{dim}": np.eye(dim)
    }

    # Step 3: Create correlation terms for each pair
    terms = []
    for site1, site2 in zip(current_sites, neighbor_sites):
        node_id1 = f"Site({site1[0]},{site1[1]})"
        node_id2 = f"Site({site2[0]},{site2[1]})"
        # Add b^†_i b_j term
        terms.append(TensorProduct({node_id1: "b^dagger", node_id2: "b"}))


    return Hamiltonian(terms, conversion_dict)

def density_density_correlation_function(Lx, Ly, dist, dim, mode="HV"):
    # <n_i * n_j> , 
    # <n_i> and <n_j> are not condidered
    if mode == "HV":
        current_sites, neighbor_sites = get_neighbors_with_distance_HV(Lx, Ly, dist)
    elif mode == "HDV":
        current_sites, neighbor_sites = get_neighbors_with_distance_HDV(Lx, Ly, dist)

    # Define operators
    _, _, number_op = bosonic_operators(dim)  # Only need number_op for density correlation
    conversion_dict = {
        "n": number_op,
        f"I{dim}": np.eye(dim)
    }

    # Create correlation terms for each pair of sites
    terms = []
    for site1, site2 in zip(current_sites, neighbor_sites):
        node_id1 = f"Site({site1[0]},{site1[1]})"
        node_id2 = f"Site({site2[0]},{site2[1]})"
        # Add n_i * n_j term
        terms.append(TensorProduct({node_id1: "n", node_id2: "n"}))

    return Hamiltonian(terms, conversion_dict)

def Number_op_total(Lx, Ly, dim=2):
    number_op = bosonic_operators(dim)[2]
    conversion_dict = {"n": number_op , f"I{dim}": np.eye(dim)}

    terms = []
    for x in range(Lx):
        for y in range(Ly):
            current_site = f"Site({x},{y})"
            terms.append(TensorProduct({current_site: "n"}))
    return Hamiltonian(terms, conversion_dict)
 
def Number_op_local(node_id, dim=2):
    number_op = bosonic_operators(dim)[2]
    conversion_dict = {"n": number_op , f"I{dim}": np.eye(dim)}
    term = TensorProduct({node_id: "n"})
    return Hamiltonian(term, conversion_dict)       

def Correlation_function(node_id1, node_id2, dim=2):
    creation_op, annihilation_op, number_op = bosonic_operators(dim)
    conversion_dict = {
        "b^dagger": creation_op,
        "b": annihilation_op,
        f"I{dim}": np.eye(dim)
    }
    
    terms = []
    terms.append(TensorProduct({node_id1: "b^dagger", node_id2: "b"}))
    return Hamiltonian(terms, conversion_dict)

def Random_op(ttn, Lx, Ly, dim , seed = 0):
    np.random.seed(seed)
    possible_operators = [crandn((dim,dim)) for _ in range(len(ttn.nodes.keys()))] 
    conversion_dict = {f"I{dim}": np.eye(dim)}
    for i, node_id in enumerate(ttn.nodes.keys()):
        conversion_dict[node_id] = possible_operators[i]    

    terms = [TensorProduct({f"Site({x},{y})": f"Site({x},{y})" 
                                for x in range(Lx) for y in range(Ly)})]
    return Hamiltonian(terms, conversion_dict)   

