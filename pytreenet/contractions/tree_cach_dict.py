from __future__ import annotations
from typing import Union, Dict, Tuple

import numpy as np

class PartialTreeCachDict(dict):
    """
    Basically a dictionary to save the chached partial trees during TDVP.
     However, the trees are saved by a specific convention:
     The keys are tuples of size two. The first entry is the identifier of the
     node itself and the second key is the identifier of the node to which the
     open legs point
    """

    def __init__(self,
                 dictionary: Union[Dict[Tuple[str,str]],None] = None) -> None:
        if dictionary is None:
            dictionary = {}
        super().__init__(dictionary)

    def get_entry(self, node_id: str, next_node_id: str) -> np.ndarray:
        """
        Returns the cached partial tree tensor that ends at the node with
         identifier node_id and has its open legs point to the node with
         identifier next_node_id.

        Args:
            node_id (str): The identifier where the partial tree ends.
            next_node_id (str): The identifier to which the open legs point.

        Returns:
            np.ndarray: The corresponding partial tree tensor.
        """
        return self[node_id, next_node_id]

    def add_entry(self, node_id: str, next_node_id: str,
                  cached_tensor: np.ndarray):
        """
        Sets the cached partial tree tensor that ends at the node with
         identifier node_id and has its open legs point to the node with
         identifier next_node_id.

        Args:
            node_id (str): The identifier where the partial tree ends.
            next_node_id (str): The identifier to which the open legs point.
            cached_tensor (np.ndarray): The corresponding partial tree tensor.
        """
        self[node_id, next_node_id] = cached_tensor

    def change_next_id_for_entry(self, node_id: str, old_next_id,
                                 new_next_id: str):
        """
        For a given cached tensor the identifier to which the open legs of the
         chached tensor point are changed, both in the key and in the actual
         PartialTreeCache.

        Args:
            node_id (str): The node to which the entry tensor correpsonds.
            old_next_id (str): The old identifier of the node to which the
             open legs point.
            new_next_id (str): The new identifier of the node to which the
             open legs point.
        """
        chached_tensor = self.pop((node_id, old_next_id))
        self.add_entry(node_id, new_next_id, chached_tensor)

    def delete_entry(self, node_id: str, next_node_id: str):
        """
        Deletes the entry with the given node_id and next_node_id.

        Args:
            node_id (str): The identifier where the partial tree ends.
            next_node_id (str): The identifier to which the open legs point.
        """
        del self[node_id, next_node_id]
