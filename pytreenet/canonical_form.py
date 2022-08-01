import numpy as np

def canonical_form(tree_tensor_network, orthogonality_center_id):
    """

    Parameters
    ----------
    tree_tensor_network : TreeTensorNetwork
        The TTN for which to find the canonical form
    orthogonality_center_id : str
        The id of the tensor node, which sould be the orthogonality center for
        the canonical form

    Returns
    -------
    None.

    """

    distance_dict = tree_tensor_network.distance_to_node(orthogonality_center_id)

    minimum_distance = min(distance_dict.values())
    maximum_distance = min(distance_dict.values())

    for distance in range(minimum_distance, maximum_distance+1):
        pass

