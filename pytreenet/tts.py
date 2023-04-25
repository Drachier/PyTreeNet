
from .ttn import *

class TreeTensorState(TreeTensorNetwork):
    def __init__(self, original_tree, deep=True):
        super().__init__(original_tree, deep)
        self.physical_legs = dict()
        for node_id in self.nodes:
            assert len(self.nodes[node_id].open_legs) <= 1
            if len(self.nodes[node_id].open_legs) == 1:
                self.physical_legs[node_id] = self.nodes[node_id].open_legs[0]

    def adjoint(self):
        for node_id in self.nodes:
            self.nodes[node_id]._tensor = self.nodes[node_id]._tensor.conj()