"""
This module provides the PartialTreeCachDict class.

This class is a dictionary that can in general be used to save tensors that
depend on two nodes of a tree. The keys are the tuples of the two identifiers.
The main use case is to save the contraction results of subtrees during larger
contraction operations. For example during the TDVP algorithm or when computing
the expecation value of a TTNO with respect to a given TTNS.
"""
from __future__ import annotations
from typing import Union, Dict, Tuple

import numpy as np
from numpy import allclose

class PartialTreeCachDict(dict):
    """
    A dictionary to save the chached subtrees during contractions.

    The main use case is to save the contraction results of subtrees during
    larger contraction operations. For example during the TDVP algorithm or when
    computing the expecation value of a TTNO with respect to a given TTNS.

    The keys are tuples of size two. The first entry is usually the identifier
    of a node that acts as the root of a subtree that was contracted. The second
    identifier is the identifier of the node to which the open legs of the
    contracted subtree point.
    """

    def __init__(self,
                 dictionary: Union[Dict[Tuple[str,str]],None] = None) -> None:
        """
        Initializes the PartialTreeCachDict.

        Args:
            dictionary (Union[Dict[Tuple[str,str]],None], optional): A
             dictionary that contains the initial entries. Defaults to None.
        """
        if dictionary is None:
            dictionary = {}
        super().__init__(dictionary)

    def get_entry(self, node_id: str, next_node_id: str) -> np.ndarray:
        """
        Returns the cached tensor saved.

        Basically a wrapper for the __getitem__ method to ensure the correct
        order of keys.

        Args:
            node_id (str): The identifier where the subtree tree ends.
            next_node_id (str): The identifier to which the open legs point.

        Returns:
            np.ndarray: The corresponding contracted subtreetree tensor.
        """
        return self[node_id, next_node_id]

    def add_entry(self, node_id: str, next_node_id: str,
                  cached_tensor: np.ndarray):
        """
        Saves a tensor in the dictionary.

        Basically a wrapper for the __setitem__ method to ensure the correct
        order of keys.

        Args:
            node_id (str): The identifier where the subtree ends.
            next_node_id (str): The identifier to which the open legs point.
            cached_tensor (np.ndarray): The corresponding  tensor.
        """
        self[node_id, next_node_id] = cached_tensor

    def change_next_id_for_entry(self, node_id: str, old_next_id,
                                 new_next_id: str):
        """
        Canges the key for a given cached tensor.

        For a given cached tensor the identifier to which the open legs of the
        chached tensor point are changed.

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
        Deletes an entry in the dictionary.

        Basically a wrapper for the __delitem__ method to ensure the correct
        order of keys.

        Args:
            node_id (str): The identifier where the subtree ends.
            next_node_id (str): The identifier to which the open legs point.
        """
        del self[node_id, next_node_id]

    def contains(self, node_id: str, next_node_id: str) -> bool:
        """
        Checks if the dictionary contains a given entry.
        """
        return super().__contains__((node_id, next_node_id))

    def close_to(self, other: PartialTreeCachDict) -> bool:
        """
        Checks if the other cache is close to this cache.

        Args:
            other (SandwichCache): The other cache to compare with.

        Returns:
            bool: True if the other cache is close to this cache.
        """
        if len(self) != len(other):
            return False
        for key in self:
            if key not in other:
                return False
            if not allclose(self[key], other[key]):
                return False
        return True
