from __future__ import annotations
from typing import Union, Dict, Tuple

class PartialTreeChachDict(dict):
    """
    Basically a dictionary to save the chached partial trees during TDVP.
     However, the trees are saved by a specific convention:
     The keys are tuples of size two. The first entry is the identifier of the
     node itself and the second key is the identifier of the node to which the
     open legs point
    """

    def __init__(self,
                 dictionary: Union[Dict[Tuple[str,str],PartialTreeChache],None] = None) -> None:
        if dictionary is None:
            self.dictionary = {}
        else:
            self.dictionary = dictionary
            super().__init__(dictionary)

    def get_entry(self, node_id: str, next_node_id: str) -> PartialTreeChache:
        """
        Returns the cached partial tree tensor that ends at the node with
         identifier node_id and has its open legs point to the node with
         identifier next_node_id.

        Args:
            node_id (str): The identifier where the partial tree ends.
            next_node_id (str): The identifier to which the open legs point.

        Returns:
            PartialTreeChache: The corresponding partial tree tensor.
        """
        return self.dictionary[node_id, next_node_id]

    def add_entry(self, node_id: str, next_node_id: str,
                  cached_tensor: PartialTreeChach):
        """
        Sets the cached partial tree tensor that ends at the node with
         identifier node_id and has its open legs point to the node with
         identifier next_node_id.

        Args:
            node_id (str): The identifier where the partial tree ends.
            next_node_id (str): The identifier to which the open legs point.
            PartialTreeChache: The corresponding partial tree tensor.
        """
        self.dictionary[node_id, next_node_id] = cached_tensor
