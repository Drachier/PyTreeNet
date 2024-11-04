from copy import deepcopy
import math
import random
import numpy as np

from ...operators.common_operators import bosonic_operators , pauli_matrices
from ...operators import Hamiltonian, TensorProduct


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
                              seed = None): 
    Lx = len(ttn.nodes.keys()) // 3 
    Ly = Lx
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