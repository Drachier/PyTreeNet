from .ttn import *

class TreeTensorOperator(TreeTensorNetwork):
    """
    A tree tensor operator (TTO) ...
    2 open legs: index 0 for ket, index -1 for bra
    """
    def __init__(self, original_tree=None, deep=False):
        super().__init__(original_tree, deep)
        self._links = dict()

    @property
    def links(self):
        return self._links

    def add_link(self, operator_node, state_node):
        self._links.update({state_node: operator_node})
