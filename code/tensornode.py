import numpy as np
import treelib as tl

class tensornode(tl.Node):
    """
    Subclasses the Node object by adding a tensor associated to the node and
    determining the open legs and the legs to the parent and children.
    """
    
    def __init__(self, tensor, tag=None, identifier=None, expanded=True, data=None):
        
        self._tensor = tensor
        super().__init__(tag=tag, identifier=identifier, expanded=expanded, data=data)
        
        self._open_legs = list(np.arange(tensor.ndim))
        self._parent_leg = None
        self._children_legs = dict()
        
        @property
        def tensor(self):
            """
            The tensor in form of a numpy array associated to a tensornode.
            """
            return self._tensor
        
        @property
        def open_legs(self):
            """
            The tensor legs not contracted with a different tensor leg
            """
            return self._open_legs
        
        @property
        def parent_leg(self):
            """
            Leg contracted with a parent node's tensor, if existent
            """
            return self._parent_leg
        
        @property
        def children_leg(self):
            """
            The legs contracted with the children's tensors.
            The dictionary contains the children's identifier as key and the
            contracted leg as value.
            """
            return self.children_leg
        
        def open_leg_to_parent(self, open_leg_index):
            
            self.parent_leg = open_leg_index
            self._open_legs.remove(open_leg_index)
            
        def open_legs_to_children(self, open_leg_list, identifier_list):
            
            assert len(open_leg_list) == len(identifier_list)
            
            new_children_legs = dict(zip(identifier_list, open_leg_list))
            self._children_legs.update(new_children_legs)
            self.open_legs = [open_leg for open_leg in self._open_legs if open_leg not in open_leg_list]
