"""
Provides random TTNOs for testing purposes.
"""

from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator
from pytreenet.random.random_node import random_tensor_node

def generate_single_site_ttno() -> TreeTensorNetworkOperator:
    """
    Generates a single site TTNO.

    Returns:
        TreeTensorNetworkOperator: The generated TTNO.

    """
    ttno = TreeTensorNetworkOperator()
    root_id = "child_ly1"
    root_node, root_tensor = random_tensor_node((3,3),
                                     identifier=root_id)
    ttno.add_root(root_node, root_tensor)
    return ttno

def generate_three_layer_ttno() -> TreeTensorNetworkOperator:
    """
    Generates a TTNO with three layers.
    
    Returns:
        TreeTensorNetworkOperator: The generated TTNO.

                        4
                    K0------K13---2
                5  5/ \\6 
                --K10   K11
                  /4     |2
                B20
                |2
    """
    ttno = TreeTensorNetworkOperator()
    root_id = "child_ly0"
    root_node, root_tensor = random_tensor_node((4,7,6,1,1),
                                                identifier=root_id)
    ttno.add_root(root_node, root_tensor)
    child_ly1_ids = ["child_ly1" + str(i) for i in range(3)]
    node, tensor = random_tensor_node((4,2,2),
                                      identifier=child_ly1_ids[2])
    ttno.add_child_to_parent(node, tensor, 0,
                             root_id, 0)
    node, tensor = random_tensor_node((6,2,2),
                                      identifier=child_ly1_ids[1])
    ttno.add_child_to_parent(node, tensor, 0,
                             root_id, 2)
    node, tensor = random_tensor_node((7,4,5,5),
                                      identifier=child_ly1_ids[0])
    ttno.add_child_to_parent(node, tensor, 0,
                             root_id, 2)
    child_ly2_id = "child_ly2"
    node, tensor = random_tensor_node((4,2,2),
                                      identifier=child_ly2_id)
    ttno.add_child_to_parent(node, tensor, 0,
                             child_ly1_ids[0], 1)
    return ttno
