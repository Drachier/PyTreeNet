import numpy as np
from .canonical_form import *
from .tts import *
from .tensor_util import *

from scipy.linalg import expm

"""
Implements the time-dependent variational principle TDVP for tree tensor
networks.

Reference:
    D. Bauernfeind, M. Aichhorn; "Time Dependent Variational Principle for Tree
    Tensor Networks", DOI: 10.21468/SciPostPhys.8.2.024
"""

def tdvp(state, hamiltonian, time_step_size, final_time, mode="1site"):
    """

    Parameters
    ----------
    state : TreeTensorState
        The TTN representing the intial state on which the TDVP is to be 
        performed.
    hamiltonian: TreeTensorOperator
        A TTN representing the model Hamiltonian. Each tensor in
        tree_tensor_network should be associated to one node in hamiltonian.
    time_step_size : float
        Size of each time-step in the trotterisation.
    final_time : float
        Total time for which TDVP should be run.
    mode : str, optional
        Decides which version of the TDVP is run. The options are 1site and
        2site. The default is "1site".

    Returns
    -------
    state: TreeTensorState
        The TTN representing the time evolved state is returned.

    """
    assert len(state.nodes) == len(hamiltonian.nodes)

    if mode == "1site":        
        time = 0

        tdvp_update_path = find_tdvp_update_path(state)
        tdvp_orthogonalization_path = find_tdvp_orthogonalization_path(state, tdvp_update_path)
        tdvp_path = (tdvp_update_path, tdvp_orthogonalization_path)
        
        while time < final_time:
            state = time_step(state, hamiltonian, time_step_size, tdvp_path)
            time += time_step_size

    elif mode == "2site":
        raise NotImplementedError
        # TODO: Implement
    else:
        raise ValueError("Mode does not exist.")

    return state


def time_step(state, hamiltonian, dt, tdvp_path):
    canonical_form(state, tdvp_path[0][0])
    for i in range(len(tdvp_path[0])-1):
        node = tdvp_path[0][i]
        node_next = tdvp_path[0][i+1]

        state = tdvp_update(state, hamiltonian, dt, node, node_next)
        state = move_orthogonalization_center(state, node, tdvp_path) # TODO
    state = tdvp_update(state, hamiltonian, dt, tdvp_path[0][-1], end_point=True)
    return state

def tdvp_update(state, hamiltonian, dt, node, node_next, end_point=False):
    cached_sites = contract_sites(state, hamiltonian) # move up by 1 lvl in hierachy

    site_hamiltonian_effective = get_effective_site_hamiltonian(state, hamiltonian, node, cached_sites)

    half_dims = int(len(site_hamiltonian_effective.shape)/2)
    site_hamiltonian_effective_matrix = tensor_matricization(site_hamiltonian_effective, [i for i in range(half_dims)], [i+half_dims for i in range(half_dims)])
    
    # would scipys eigsh really be faster ? diagonalization takes time ...
    heff_exponential = expm(-1j * site_hamiltonian_effective_matrix * dt)

    site_tensor_vector = np.reshape(state.nodes[node].tensor, newshape=heff_exponential.shape[0])
    site_tensor_updated = heff_exponential @ site_tensor_vector

    link_dim = state.nodes[node].tensor.shape[-1]
    site_tensor_updated_matrix = np.reshape(site_tensor_updated, newshape=(np.prod(site_tensor_updated.shape)-link_dim, link_dim))

    q, r = np.linalg.qr(site_tensor_updated_matrix)
    # TODO tensor_qr_decomposition from tensor_util
    
    state.nodes[node].tensor = q
    cached_sites[node] = contract_site(state, hamiltonian, node)

    # TODO stop if endpoint is reached
    link_hamiltonian_effective, node_next_neighbour = get_effective_link_hamiltonian(state, site_hamiltonian_effective, node, node_next, cached_sites)

    half_dims = int(len(link_hamiltonian_effective.shape)/2)
    link_hamiltonian_effective_matrix = tensor_matricization(link_hamiltonian_effective, [i for i in range(half_dims)], [i+half_dims for i in range(half_dims)])

    # would scipys eigsh really be faster ? diagonalization takes time ...
    heff_exponential = expm(+1j * link_hamiltonian_effective_matrix * dt)

    link_tensor_vector = np.reshape(r, newshape=heff_exponential.shape[0])
    link_tensor_updated = heff_exponential @ link_tensor_vector
    link_dim = state.nodes[node].tensor.shape[-1]
    link_tensor_updated_matrix = np.reshape(link_tensor_updated, newshape=(np.prod(link_tensor_updated.shape)-link_dim, link_dim))

    state.nodes[node_next_neighbour[0]].tensor = np.tensordot(link_tensor_updated_matrix, state.nodes[node_next_neighbour[0]].tensor, axes=(1, node_next_neighbour[1]))
    cached_sites[node_next_neighbour[0]] = contract_site(state, hamiltonian, node_next_neighbour[0])    
    return state


def  move_orthogonalization_center(state, node, tdvp_path=None):
    if tdvp_path is None:
        # Naive implementation
        canonical_form(state, node)
    else:
        # TODO
        canonical_form(state, node)
    return state


def contract_sites(state, hamiltonian):
    contracted_sites = dict()
    for node_id in state.nodes.keys():
        bra_h_ket = contract_site(state, hamiltonian, node_id)
        contracted_sites[node_id] = bra_h_ket
    return contracted_sites

def contract_site(state, hamiltonian, node_id):
    bra = state.nodes[node_id].tensor.T.conj()
    bra_contract_dim = state.nodes[node_id].open_legs

    h = hamiltonian.nodes[node_id].tensor
    h_bra_contract_dim = [hamiltonian.nodes[node_id].open_legs[1]]
    h_ket_contract_dim = [hamiltonian.nodes[node_id].open_legs[0]]

    ket = state.nodes[node_id].tensor
    ket_contract_dim = state.nodes[node_id].open_legs

    bra_h = np.tensordot(bra, h, axes=(bra_contract_dim, h_bra_contract_dim))
    bra_h_ket = np.tensordot(bra_h, ket, axes=(h_ket_contract_dim, ket_contract_dim))
    return bra_h_ket

def get_effective_site_hamiltonian(state, hamiltonian, node, cached_sites):
    eff_hamiltonian = hamiltonian.nodes[node].tensor

    if node != state.root_id:
        parent_part = cached_sites[state.root_id]
        parent_part = absorb_children_except(state, cached_sites, state.root_id, parent_part, node)
        eff_hamiltonian = np.tensordot(parent_part, eff_hamiltonian, axes=(1, 0))

    for child_id in state.nodes[node].children_legs.keys():
        child_part = cached_sites[child_id]
        child_part = absorb_children_except(state, cached_sites, child_id, child_part, "absorb all")
        eff_hamiltonian = np.tensordot(eff_hamiltonian, child_part, axes=(1, 0))

    return eff_hamiltonian


def get_effective_link_hamiltonian(state, eff_site_hamiltonian, node, node_next, cached_sites):
    if node != state.root_id and node_next == state.nodes[node].parent_leg[0]:
        node_id_of_next_node = state.nodes[node].parent_leg[1]
        node_next = [node_next]
        node_next.append(state.nodes[node_next[0]].children_legs[node])
    elif node_next in state.nodes[node].children_legs.keys():
        node_id_of_next_node = state.nodes[node].children_legs[node_next]
        node_next = [node_next]
        node_next.append(0)
    else:
        node_id_of_next_node = state.nodes[node].parent_leg[1]
        node_next = [state.nodes[node].parent_leg[0]]
        node_next.append(state.nodes[node_next[0]].children_legs[node])


    site_tensor_dims = len(state.nodes[node].tensor.shape)
    list_of_nodes_except_next_node = [i for i in range(site_tensor_dims) if i != node_id_of_next_node]
    list_of_nodes_except_next_node_incremented_by_one = [i+1 for i in list_of_nodes_except_next_node]

    eff_site_hamiltonian = np.tensordot(eff_site_hamiltonian, state.nodes[node].tensor, axes=(list_of_nodes_except_next_node, list_of_nodes_except_next_node))
    eff_site_hamiltonian = np.tensordot(eff_site_hamiltonian, state.nodes[node].tensor.T.conj(), axes=(list_of_nodes_except_next_node_incremented_by_one, list_of_nodes_except_next_node))
    return eff_site_hamiltonian, node_next


def absorb_children_except(state, cached_sites, parent_id, tensor, node):
    for child_id in state.nodes[parent_id].children_legs.keys():
        if child_id != node:
            parent_dims = np.array([0, 1, 2]) * len(tensor.shape)/3
            child_dims = np.array([0, 1, 2]) * len(cached_sites[child_id].shape)/3
            tensor = np.tensordot(tensor, cached_sites[child_id], axes=(parent_dims.astype(np.int32), child_dims.astype(np.int32)))
            tensor = absorb_children_except(state, cached_sites, child_id, tensor, node)
    return tensor


def find_tdvp_update_path(state, mode="optimized"):
    path = [i for i in state.nodes.keys()]

    if mode=="shuffle":
        path_ = path
        count = count_path_orthogonalizations(path, state)
        for i in range(4269):
            np.random.shuffle(path_)
            count_ = count_path_orthogonalizations(path_, state)
            if count_ < count:
                path = path_
                count = count_
    elif mode=="optimized":
        # Start with leaf furthest from root.
        distances_from_root = state.distance_to_node(state.root_id)
        start = max(distances_from_root, key=distances_from_root.get)

        # Move from start to root. Start might not be exactly start, but another leaf. 
        sub_path = find_tdvp_path_from_leaves_to_root(state, start)
        path = [] + sub_path + [state.root_id]
        
        branch_roots = [x for x in state.nodes[state.root_id].children_legs.keys() if x not in path]

        # TODO 2. längsten branch zum schluss bitte

        for branch_root in branch_roots:
            sub_path = find_tdvp_path_from_leaves_to_root(state, branch_root)
            sub_path.reverse()
            path = path + sub_path
    return path


def find_tdvp_path_from_leaves_to_root(state, any_child):
    path_from_child_to_root = find_path_to_root(state, any_child)
    branch_origin = path_from_child_to_root[-2]

    path = []
    path = find_tdvp_path_for_branch(state, branch_origin, path)
    return path


def find_tdvp_path_for_branch(state, branch_origin, path):
    node = branch_origin
    children = state.nodes[node].children_legs.keys()
    for child in children:
        path = find_tdvp_path_for_branch(state, child, path)
    path.append(node)
    return path


def count_path_orthogonalizations(path, ttn):
    count = 0

    # create initial orthogonalization center
    count += len(path) - 1

    previous_node = path[0]
    for node in path[1::]:
        count += distance_from_previous_node(ttn, node, previous_node)
        previous_node = node
    return count


def distance_from_previous_node(ttn, node, previous_node):
    distance = ttn.distance_to_node(previous_node)[node]
    return distance


def find_tdvp_orthogonalization_path(state, tdvp_update_path):
    path = []
    cache_sub_path_next_center = []

    # Initial canonucal form is provided by pytreenet.canonical_form(state, tdvp_update_path[0]).
    # This function provides an order of orthogonalizations to be performed - so canonical_form() does not have to be called every tdvp step (overkill)
    for i in range(len(tdvp_update_path)-1):
        # For current orthonomal center and next orthonormal center:
        # Write path from node to root. Inverse for next orthonormal center. Remove duplicates in both. Remove the second node, keep the first, keep root only once.

        if cache_sub_path_next_center != []:
            sub_path_current_center = cache_sub_path_next_center
        else:
            sub_path_current_center = find_path_to_root(state, tdvp_update_path[i])
        sub_path_next_center = find_path_to_root(state, tdvp_update_path[i+1])
        cache_sub_path_next_center = sub_path_next_center

        sub_path = sub_path_current_center + sub_path_next_center[::-1]

        # Duplicate node might either be passed on the way from one branch to another (remove once) or from one branch back to itself (remove both occurences).
        # Reverse ordering of duplicates to start with root/parents.
        duplicates = [j for j in sub_path if sub_path.count(j) != 1]
        duplicates = duplicates[0:len(duplicates)//2]
        duplicates.reverse()
        for node in duplicates:
            if len(set([j for j in sub_path if j in state.nodes[node].children_legs.keys()])) > 1: # Count no. of different children.
                # Two different branches: Remove once.
                sub_path_new = []
                [sub_path_new.append(j) for j in sub_path if j not in sub_path_new or j != node]
                sub_path = sub_path_new
            else:
                # Same branch: Remove both.
                sub_path = [j for j in sub_path if j != node]

        if tdvp_update_path[i] not in sub_path:
            # Root is start node and got removed.
            sub_path = [tdvp_update_path[i]] + sub_path
        if sub_path == []: # Next center is child of current center.
            sub_path.append(tdvp_update_path[i])

        # If target is not parent (of parent of ...) of start, then target escapes duplicate removal and needs to be removed manually.
        if sub_path[-1] == tdvp_update_path[i+1]:
            path.append(sub_path[:-1])
        else:
            path.append(sub_path)
    return path


def find_path_to_root(state, node):
    path = [node] # Startinbg point.
    while not state.nodes[node].is_root():
        path.append(state.nodes[node].parent_leg[0])
        node = state.nodes[node].parent_leg[0]
    return path
