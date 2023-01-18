from ttn import TreeTensorNetwork

def TreeTensorState(TreeTensorNetwork):
    def __init__(self, original_tree = None, deep = False):

        if original_tree is not None:
            for node_id in original_tree.nodes:
                # Test that number of open legs is either 1 or 0.
                # In the latter case add one trivial leg, i.e. of dim 1.