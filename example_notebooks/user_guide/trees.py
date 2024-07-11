from typing import Tuple, Dict

import pytreenet as ptn

def construct_orig_pub_tree() -> Tuple[ptn.TreeTensorNetworkState,Dict[str,int]]:
    """
    Constructs the tree tensor network used in the original publication on
    TTNO and state diagrams.
    """
    ttns = ptn.TreeTensorNetworkState()
    # Physical legs come last
    node1, tensor1 = ptn.random_tensor_node((1, 1, 2), identifier="site1")
    node2, tensor2 = ptn.random_tensor_node((1, 1, 1, 2), identifier="site2")
    node3, tensor3 = ptn.random_tensor_node((1, 2), identifier="site3")
    node4, tensor4 = ptn.random_tensor_node((1, 2), identifier="site4")
    node5, tensor5 = ptn.random_tensor_node((1, 1, 1, 2), identifier="site5")
    node6, tensor6 = ptn.random_tensor_node((1, 2), identifier="site6")
    node7, tensor7 = ptn.random_tensor_node((1, 1, 2), identifier="site7")
    node8, tensor8 = ptn.random_tensor_node((1, 2), identifier="site8")
    ttns.add_root(node1, tensor1)
    ttns.add_child_to_parent(node2, tensor2, 0, "site1", 0)
    ttns.add_child_to_parent(node3, tensor3, 0, "site2", 1)
    ttns.add_child_to_parent(node4, tensor4, 0, "site2", 2)
    ttns.add_child_to_parent(node5, tensor5, 0, "site1", 1)
    ttns.add_child_to_parent(node6, tensor6, 0, "site5", 1)
    ttns.add_child_to_parent(node7, tensor7, 0, "site5", 2)
    ttns.add_child_to_parent(node8, tensor8, 0, "site7", 1)

    leg_dict = {"site1": 0, "site2": 1, "site3": 2, "site4": 3, "site5": 4,
                 "site6": 5, "site7": 6, "site8": 7}
    return ttns, leg_dict

def construct_T_tree() -> Tuple[ptn.TreeTensorNetworkState,Dict[str,int]]:
    """
    Constructs the T shaped TTNS used in the userguide to demonstrate TTNS.

    However, to simplify the random generation, all physical legs have
    dimension 2.
    """
    ttns = ptn.TreeTensorNetworkState()
    center_node = ptn.Node(identifier="0")
    center_tensor = ptn.std_utils.crandn((4,4,4,2))
    ttns.add_root(center_node, center_tensor)
    for i in range(3):
        chain_node = ptn.Node(identifier=f"{i}0")
        chain_tensor = ptn.std_utils.crandn((4,3,2))
        ttns.add_child_to_parent(chain_node, chain_tensor,
                                    0,"0",i)
        end_node = ptn.Node(identifier=f"{i}1")
        end_tensor = ptn.std_utils.crandn((3,2))
        ttns.add_child_to_parent(end_node, end_tensor,
                                    0,f"{i}0",1)
    leg_dict = {"0": 0, "00": 1, "01": 2, "10": 3, "11": 4, "20": 5, "21": 6}
    return ttns, leg_dict

def construct_user_guide_example_tree() -> Tuple[ptn.TreeTensorNetworkState,Dict[str,int]]:
    """
    Constructs a TTNS with the same structure as the example in the userguide,
    but where all nodes have an open leg of dimension 2.
    """
    ttn = ptn.TreeTensorNetwork()
    root_node, root_tensor = ptn.random.random_tensor_node((2,4,5,3),
                                                        "root")
    ttn.add_root(root_node, root_tensor)
    node1, tensor1 = ptn.random.random_tensor_node((4,2,2,2),"node1")
    ttn.add_child_to_parent(node1, tensor1, 0, "root", 1)
    node4, tensor4 = ptn.random.random_tensor_node((5,2),"node4")
    node5, tensor5 = ptn.random.random_tensor_node((3,2,2),"node5")
    ttn.add_child_to_parent(node4, tensor4, 0, "root", 2)
    ttn.add_child_to_parent(node5, tensor5, 0, "root", 3)
    node2, tensor2 = ptn.random.random_tensor_node((2,2),"node2")
    node3, tensor3 = ptn.random.random_tensor_node((2,2),"node3")
    node6, tensor6 = ptn.random.random_tensor_node((2,2),"node6")
    ttn.add_child_to_parent(node2, tensor2, 0, "node1", 1)
    ttn.add_child_to_parent(node3, tensor3, 0, "node1", 2)
    ttn.add_child_to_parent(node6, tensor6, 0, "node5", 1)

    leg_dict = {"root": 0, "node1": 1, "node4": 2, "node5": 3, "node2": 4,
                "node3": 5, "node6": 6}
    return ttn, leg_dict
