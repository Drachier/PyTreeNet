from ...contractions.contraction_util import determine_index_with_ignored_leg
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...time_evolution.Subspace_expansion import contract_ttno_with_ttno 
from ...operators.tensorproduct import TensorProduct
from ...operators.hamiltonian import Hamiltonian
from ...operators.common_operators import bosonic_operators


from copy import deepcopy
import numpy as np

def contract_vectorized_rho_to_ttno(vectorized_rho):  
    vectorized_rho_ttno = deepcopy(vectorized_rho)  
    for ket_id in [node.identifier for node in vectorized_rho_ttno.nodes.values() if str(node.identifier).startswith("S")]:
            bra_id = ket_id.replace("Site", "Node")
            rho_id = ket_id.replace("Site", "Vertex")
            vectorized_rho_ttno.contract_nodes(ket_id, bra_id, rho_id)
    return vectorized_rho_ttno  

def trace_leaf(tensor):
    return np.trace(tensor, axis1=-2, axis2=-1)

def trace_subtrees_using_dictionary(parent_id, node, tensor, dictionary): 
    node_id = node.identifier
    result_tensor = tensor
    for neighbour_id in node.neighbouring_nodes():
        if neighbour_id != parent_id:
           tensor_index_to_neighbour = determine_index_with_ignored_leg(node, neighbour_id, parent_id)
           cached_neighbour_tensor = dictionary.get_entry(neighbour_id,node_id)
           result_tensor = np.tensordot(result_tensor, cached_neighbour_tensor, axes=([tensor_index_to_neighbour],[0]))
    return np.trace(result_tensor, axis1=-2, axis2=-1)

def trace_any(node_id, vectorized_rho, dictionary):
    node, tensor = vectorized_rho[node_id]
    parent_id = node.parent
    if node.is_leaf():
        return trace_leaf(tensor)
    return trace_subtrees_using_dictionary(parent_id, node, tensor, dictionary)

def contract_root_with_environment(vectorized_rho, dictionary):
    node_id = vectorized_rho.root_id
    node, tensor = vectorized_rho[node_id]
    result_tensor = tensor
    for neighbour_id in node.neighbouring_nodes():
        cached_neighbour_tensor = dictionary.get_entry(neighbour_id,node_id)
        tensor_leg_to_neighbour = node.neighbour_index(neighbour_id)
        result_tensor = np.tensordot(result_tensor, cached_neighbour_tensor, axes=([tensor_leg_to_neighbour],[0]))
    return np.trace(result_tensor, axis1=-2, axis2=-1)

def trace_vectorized_rho(ttno_rho):
    dictionary = PartialTreeCachDict()
    for node_id in ttno_rho.linearise()[:-1]:
        trace = trace_any(node_id, ttno_rho, dictionary)
        parent_id = ttno_rho.nodes[node_id].parent
        dictionary.add_entry(node_id, parent_id, trace)
        children = ttno_rho.nodes[node_id].children
        for child_id in children:
            dictionary.delete_entry(child_id,node_id)    
    return complex(contract_root_with_environment(ttno_rho,dictionary))

def expectation_value_Lindblad(vectorized_rho , operator):
    rho_ttno = contract_vectorized_rho_to_ttno(vectorized_rho)
    op_rho = contract_ttno_with_ttno(operator, rho_ttno)
    return trace_vectorized_rho(op_rho)  

def trace_rho_squared(vectorized_rho):
    rho_ttno = contract_vectorized_rho_to_ttno(vectorized_rho)
    rho_squared = contract_ttno_with_ttno(rho_ttno, rho_ttno)
    return trace_vectorized_rho(rho_squared)

def trace_rho(vectorized_rho):
    rho_ttno = contract_vectorized_rho_to_ttno(vectorized_rho)
    return trace_vectorized_rho(rho_ttno)



def get_neighbors_Site(x, y, Lx, Ly):
  neighbors = []
  
  # Right neighbor
  if x < Lx - 1:
      neighbors.append(f"Site({x+1},{y})")
  
  # Up neighbor
  if y < Ly - 1:
      neighbors.append(f"Site({x},{y+1})")
  
  return neighbors

def get_neighbors_Node(x, y, Lx, Ly):
  neighbors = []

  # Right neighbor
  if x < Lx - 1:
      neighbors.append(f"Node({x+1},{y})")
  
  # Up neighbor
  if y < Ly - 1:
      neighbors.append(f"Node({x},{y+1})")
  
  return neighbors

def BH_Liouville(t, U, gamma, m, L, Lx, Ly, d):
    creation_op, annihilation_op, number_op = bosonic_operators(d)
    
    conversion_dict = {
        "b^dagger": creation_op,
        "b": annihilation_op,
        f"I{d}": np.eye(d)
    }
    
    conversion_dict.update({
        "it * b^dagger": t*1j * creation_op,
        "it * b": t*1j * annihilation_op,
        "-iU * n * (n - 1)": -U*1j * number_op @ (number_op - np.eye(d)),
        "im*n": m*1j*number_op
    })
    
    terms = []
    
    # Hopping terms
    for x in range(Lx):
        for y in range(Ly):
            current_site = f"Site({x},{y})"
            neighbors = get_neighbors_Site(x, y, Lx, Ly)            
            for neighbor in neighbors:
                terms.append(TensorProduct({current_site: "it * b^dagger", neighbor: "b"}))
                terms.append(TensorProduct({current_site: "it * b", neighbor: "b^dagger"}))
                

    
    # On-site interaction terms
    for x in range(Lx):
        for y in range(Ly):
            current_site = f"Site({x},{y})"
            terms.append(TensorProduct({current_site: "-iU * n * (n - 1)"}))

    # Chemical potential terms
    for x in range(Lx):
        for y in range(Ly):
            current_site = f"Site({x},{y})"
            terms.append(TensorProduct({current_site: "im*n"}))        
    
    H1 = Hamiltonian(terms, conversion_dict)
    
    conversion_dict = {
        "b^dagger.T": creation_op.T,
        "b.T": annihilation_op.T,
        f"I{d}": np.eye(d)
    }
    
    conversion_dict.update({
        "-it * b^dagger.T": -t*1j * creation_op.T,
        "-it * b.T": -t*1j * annihilation_op.T,
        "iU * n * (n - 1).T": (U*1j * number_op @ (number_op - np.eye(d))).T,
        "-im*n.T": -m*1j* number_op.T
    })
    
    terms = []
    
    # Hopping terms
    for x in range(Lx):
        for y in range(Ly):
            current_site = f"Node({x},{y})"
            neighbors = get_neighbors_Node(x, y, Lx, Ly)
            for neighbor in neighbors:
                terms.append(TensorProduct({current_site: "-it * b^dagger.T", neighbor: "b.T"}))
                terms.append(TensorProduct({current_site: "-it * b.T", neighbor: "b^dagger.T"}))

    # On-site interaction terms
    for x in range(Lx):
        for y in range(Ly):
            current_site = f"Node({x},{y})"
            terms.append(TensorProduct({current_site: "iU * n * (n - 1).T"}))    

    # Chemical potential terms
    for x in range(Lx):
        for y in range(Ly):
            current_site = f"Node({x},{y})"
            terms.append(TensorProduct({current_site: "-im*n.T"}))
            
    H2 = Hamiltonian(terms, conversion_dict)
    H1.__add__(H2)

        
    conversion_dict = {    
    "L": np.sqrt(gamma) * L,
    "L^dagger.T": np.sqrt(gamma) * L.conj(),
    "-1/2 (L^dagger @ L) " : -1/2 * gamma * L.conj().T @ L,
    "-1/2 (L^dagger @ L).T": -1/2 * gamma * (L.conj().T @ L).T}
    terms = []
    for x in range(Lx):
        for y in range(Ly):
            out_site = f"Node({x},{y})"
            in_site = f"Site({x},{y})"
            terms.append(TensorProduct({in_site: "L" , out_site: "L^dagger.T"}))
            terms.append(TensorProduct({in_site: "-1/2 (L^dagger @ L) "}))
            terms.append(TensorProduct({out_site: "-1/2 (L^dagger @ L).T"}))

    H3 = Hamiltonian(terms, conversion_dict)
    H1.__add__(H3)
    return H1


# get the ttno_rho and replace all tensors with zeros with one physical leg
# to be used to generate operator in with TTNO.from_hamiltonian
def rho_structure(vectorized_rho):
    ttno_rho = contract_vectorized_rho_to_ttno(vectorized_rho)
    for node_id in ttno_rho.nodes.keys():
        shape = ttno_rho.tensors[node_id].shape
        new_shape = tuple([1] * (len(shape) -2) + [2])
        I = np.zeros(new_shape)
        ttno_rho.tensors[node_id] = I
        ttno_rho.nodes[node_id].link_tensor(I)
    return ttno_rho    
 