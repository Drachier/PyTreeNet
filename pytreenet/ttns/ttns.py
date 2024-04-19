from __future__ import annotations
from typing import List, Union, Any, Dict
from copy import deepcopy
from copy import copy

import numpy as np

from ..core.ttn import TreeTensorNetwork
from ..operators.tensorproduct import TensorProduct
from ..contractions.state_state_contraction import contract_two_ttns
from ..util import copy_object
from ..util import compute_transfer_tensor
from ..util.tensor_splitting import SVDParameters
from ..time_evolution.time_evo_util.update_path import TDVPUpdatePathFinder
from ..ttno import TTNO
from ..core.canonical_form import complete_canonical_form

class TreeTensorNetworkState(TreeTensorNetwork):
    """
    This class holds methods commonly used with tree tensor networks
     representing a state.
    """ 
    
    def scalar_product(self) -> complex:
        """
        Computes the scalar product of this TTNS

        Returns:
            complex: The resulting scalar product <TTNS|TTNS>
        """
        if self.orthogonality_center_id is not None:
            tensor = self.tensors[self.orthogonality_center_id]
            tensor_conj = tensor.conj()
            legs = tuple(range(tensor.ndim))
            return complex(np.tensordot(tensor, tensor_conj, axes=(legs,legs)))
        # Very inefficient, fix later without copy
        ttn = deepcopy(self)
        return contract_two_ttns(ttn, ttn.conjugate())

    def single_site_operator_expectation_value(self, node_id: str,
                                               operator: np.ndarray) -> complex:
        """
        Find the expectation value of this TTNS given the single-site operator acting on
         the node specified.
        Assumes the node has only one open leg.

        Args:
            node_id (str): The identifier of the node, the operator is applied to.
            operator (np.ndarray): The operator of which we determine the expectation value.
             Note that the state will be contracted with axis/leg 0 of this operator.

        Returns:
            complex: The resulting expectation value < TTNS| Operator| TTN >
        """
        if self.orthogonality_center_id == node_id:
            tensor = deepcopy(self.tensors[node_id])
            tensor_op = np.tensordot(tensor, operator, axes=(-1,0))
            tensor_conj = tensor.conj()
            legs = tuple(range(tensor.ndim))
            return complex(np.tensordot(tensor_op, tensor_conj, axes=(legs,legs)))

        tensor_product = TensorProduct({node_id: operator})
        return self.operator_expectation_value(tensor_product)

    def operator_expectation_value(self, operator: TensorProduct) -> complex:
        """
        Finds the expectation value of the operator specified, given this TTNS.

        Args:
            operator (TensorProduct): A TensorProduct representing the operator
             as many single site operators.

        Returns:
            complex: The resulting expectation value < TTNS | operator | TTNS>
        """
        # Very inefficient, fix later without copy
        ttn = deepcopy(self)
        conj_ttn = ttn.conjugate()
        for node_id, single_site_operator in operator.items():
            ttn.absorb_into_open_legs(node_id, single_site_operator)
        return contract_two_ttns(ttn, conj_ttn)

    def is_in_canonical_form(self, node_id: Union[None,str] = None) -> bool:
        """
        Returns whether the TTNS is in canonical form. If a node_id is specified,
         it will check as if that node is the orthogonalisation center. If no
         node_id is given, the current orthogonalisation center will be used.

        Args:
            node_id (Union[None,str], optional): The node to check. If None, the
             current orthogonalisation center will be used. Defaults to None.
        
        Returns:
            bool: Whether the TTNS is in canonical form.
        """
        if node_id is None:
            node_id = self.orthogonality_center_id
        if node_id is None:
            return False
        total_contraction = self.scalar_product()
        local_tensor = self.tensors[node_id]
        legs = range(local_tensor.ndim)
        local_contraction = complex(np.tensordot(local_tensor, local_tensor.conj(),
                                                 axes=(legs,legs)))
        # If the TTNS is in canonical form, the contraction of the
        # orthogonality center should be equal to the norm of the state.
        return np.allclose(total_contraction, local_contraction)
    
    def reduced_density_matrix(self , node_id: str, SVDParameters = SVDParameters() ) -> np.ndarray: 
        """Computes the reduced density matrices of a tree tensor network.
        Args:
            ttn: A tree tensor network.
            node_id: str
                     The node_id of the node for which the reduced density matrix is computed.
            max_bond_dim: int, optional
                          The maximum bond dimension of the reduced density matrices.
                          Default is np.inf.
            rel_tol: float, optional
                     The relative tolerance for the truncation of the singular values.
                     Default is -np.inf.
            total_tol: float, optional
                       The total tolerance for the truncation of the singular values.
                       Default is -np.inf.
         """    
        working_ttn = self.normalize_ttn(to_copy = True)
        working_ttn.canonical_form(node_id , SVDParameters)
        contracted_legs = tuple(range(working_ttn.tensors[node_id].ndim - 1 ))
        reduced_density = compute_transfer_tensor(working_ttn.tensors[node_id], contracted_legs)
        return reduced_density
    
    def reduced_density_matrix_dict(self, SVDParameters :SVDParameters = SVDParameters()) -> Dict[str, np.ndarray]: 
        """Computes the reduced density matrices of a tree tensor network.
        Args:
            ttn: A tree tensor network.
            max_bond_dim: int, optional
                          The maximum bond dimension of the reduced density matrices.
                          Default is np.inf.
            rel_tol: float, optional
                     The relative tolerance for the truncation of the singular values.
                     Default is -np.inf.
            total_tol: float, optional
                       The total tolerance for the truncation of the singular values.
                       Default is -np.inf.
        Returns:
            dict: A dictionary containing the reduced density matrices of the tree tensor network.
            dict[node_id]: The reduced density matrix of the node with the given node_id.
        """
    
        update_path = TDVPUpdatePathFinder(self).find_path()
        self = self.normalize_ttn(to_copy = True)
        self.canonical_form(update_path[0],SVDParameters)

        orthogonalization_path = []
        for i in range(len(update_path)-1):
            sub_path = self.path_from_to(update_path[i], update_path[i+1])
            orthogonalization_path.append(sub_path[1::])

        dict = {}
        for i, node_id in enumerate(update_path):
            contracted_legs = tuple(range(self.tensors[node_id].ndim - 1 )) 
            if i == len(update_path)-1:
                reduced_density = compute_transfer_tensor(self.tensors[node_id], contracted_legs)        
                dict[node_id] = reduced_density
            elif i == 0:
                reduced_density = compute_transfer_tensor(self.tensors[node_id], contracted_legs)        
                dict[node_id] = reduced_density
                next_node_id = orthogonalization_path[0][0]
                move_orth_for_path(self,[node_id, next_node_id], SVDParameters)
            else:
                current_orth_path = orthogonalization_path[i-1]
                move_orth_for_path(self,current_orth_path,SVDParameters)
                reduced_density = compute_transfer_tensor(self.tensors[node_id], contracted_legs)        
                dict[node_id] = reduced_density
                next_node_id = orthogonalization_path[i][0]
                move_orth_for_path(self,[node_id, next_node_id], SVDParameters)
        return dict

    def density_tensor(self) -> (np.ndarray, List[str]):
        """
        Computes the density tensor of a tree tensor network.
        Args :
            ttn: TreeTensorNetwork
        Returns :
            density tensor : Tensor , 
            order : List[str]
            # order of legs =  [out_1, out_2, ..., out_n, in_1, in_2, ..., in_n] 
        """    
        ttn_cct = self.normalize_ttn(to_copy=True)
        tensor , order = ttn_cct.completely_contract_tree()

        ket = tensor
        bra = tensor.conj()
        ket = ket.reshape(ket.shape + (1,))
        bra = bra.reshape(bra.shape + (1,))

        return  np.tensordot(bra,ket,axes=([-1],[-1])) , order
    
    def density_ttno(self):
        """
        Computes the ttno with density tensor of a ttn.
        Args :
            ttn: TreeTensorNetwork
        Returns :
            ttno : TTNO 
            order of legs =  [out_1, in_1, ..., out_n, in_n]
        """
        tensor , order = self.density_tensor()
        leg_dict = {order[i] : i for i in range(len(order))}
        return TTNO.from_tensor(self, tensor, leg_dict) 
    
    
    def reduced_density_matrix_2(self ,node_id : str) -> np.ndarray:
        """
        Computes the reduced density matrix of a node in a tree tensor network.
        Args :
            ttn: TreeTensorNetwork
            node_id: str
        Returns :
            pho : Tensor
        """
        pho , order = self.density_tensor()
        dim = pho.ndim
        for i in range(len(order)):
            if order[i] == node_id:
                count = i
        for i in range(count):
            pho = np.trace(pho, axis1 = 0, axis2 = pho.ndim//2)
        for i in range(count +1 , dim//2):
            pho = np.trace(pho, axis1 = 1, axis2 = 1 + pho.ndim//2)
        return pho
    
    def complete_canonical_form(self , SVDParameters):
        """
        Brings the TTN in canonical form with respect to the root node.
        """
        norm = complete_canonical_form(self, SVDParameters)
        return self , norm     

    def normalize_ttn(self,to_copy: bool=False) -> TreeTensorNetwork:
        return normalize_ttn(self, to_copy=to_copy)


def move_orth_for_path(ttn: TreeTensorNetwork , path: List[str] , SVDParameters):
        if len(path) == 0:
            return
        assert ttn.orthogonality_center_id == path[0]
        for i, node_id in enumerate(path[1:]):
            ttn. move_orthogonalization_center_svd( node_id, SVDParameters) 


def normalize_ttn(ttn: TreeTensorNetwork , to_copy = False):
   """
    Normalize a tree tensor network.
    Args:
        ttn : TreeTensorNetwork
        The tree tensor network to normalize.
        to_copy : bool, optional
                  If True, the input tree tensor network is not modified and a new tree tensor network is returned.
                  If False, the input tree tensor network is modified and returned.
                  Default is False.
    Returns : 
        The normalized tree tensor network.
    """
   ttn_normalized = copy_object(ttn, deep=to_copy)
   if len(ttn_normalized.nodes) == 1:
       node_id = list(ttn_normalized.nodes.keys())[0]
       tensor = ttn_normalized.tensors[node_id]
       indices  = tuple(ttn_normalized.nodes[node_id].open_legs)
       norm = np.sqrt(np.tensordot(tensor,tensor.conj(), axes = (indices , indices) ))
       ttn_normalized.tensors[node_id] = ttn_normalized.tensors[node_id] / norm
   else :    
      norm = contract_two_ttns(ttn_normalized,ttn_normalized.conjugate())
      for node_id in list(ttn_normalized.nodes.keys()):
         norm = contract_two_ttns(ttn_normalized,ttn_normalized.conjugate())
         ttn_normalized.tensors[node_id] /= np.sqrt(norm)
   return ttn_normalized
