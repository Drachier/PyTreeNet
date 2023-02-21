from .ttn import TreeTensorNetwork
from .tensor_util import tensor_qr_decomposition
from .tensornode import TensorNode


class TTNO(TreeTensorNetwork):

    def __init__(self, **kwargs):
        TreeTensorNetwork.__init__(self, **kwargs)
        
    def from_hamiltonian(self, hamiltonian):
        """
        

        Parameters
        ----------
        hamiltonian : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        raise NotImplementedError()

    def from_tensor(self, reference_tree, tensor, leg_dict):
        """


        Parameters
        ----------
        reference_tree : TreeTensorNetwork
            A tree used as a reference. The TTNO will have the same underlying
            tree structure and the same node_ids.
        tensor : ndarray
            A big numpy array that represents an operator on a lattice of
            quantum systems. Therefore it has to have an even number of legs.
        leg_dict : dict
            A dictionary it contains node_identifiers as keys and leg indices
            as values. It is used to match the legs of tensor to the different
            nodes. Only the lower half of the legs is to be put in this dict,
            the others are inferred by ordering.

        Returns
        -------
        None.

        """

        assert tensor.ndim % 2 == 0
        half_dim = int(tensor.ndim / 2)
        assert half_dim == len(reference_tree.nodes)

        root_id = reference_tree.root_id
        new_leg_dict = {node_id: [leg_dict[node_id], half_dim + leg_dict[node_id]]
                        for node_id in leg_dict}

        root_node = TensorNode(tensor, identifier=root_id)
        self.add_root(root_node)

        self._from_tensor_rec(reference_tree, root_node, new_leg_dict)

    def _from_tensor_rec(self, reference_tree, current_node, leg_dict):
        """
        Recursive part to obtain a TTNO from a tensor. For each child of the
        current node a new node is defined via a QR-decomposition and the 
        current_node and the leg_dict are modified.

        Parameters
        ----------
        reference_tree : TreeTensorNetwork
            A tree used as a reference. The TTNO will have the same underlying
            tree structure and the same node_ids.
        current_node : TensorNode
            The current node which we want to split via a QR-decomposition.
        leg_dict : dict
            A dictionary it contains node_identifiers as keys and leg indices
            as values. It is used to match the legs of tensor in the current
            node to the different nodes it will be split into.

        Returns
        -------
        None.

        """
        # At a leaf, we can immediately stop
        if reference_tree.nodes[current_node.identifier].is_leaf():
            return

        current_children = reference_tree.nodes[current_node.identifier].get_children_ids()

        for child_id in current_children:
            
            q_legs, r_legs, q_leg_dict, r_leg_dict = TTNO._prepare_legs_for_QR(reference_tree, current_node, child_id, leg_dict)

            current_tensor = current_node.tensor
            Q, R = tensor_qr_decomposition(current_tensor, q_legs, r_legs)

            q_node = TTNO._create_q_node(Q, current_node, q_leg_dict)
            new_q_leg_dict = TTNO._find_new_leg_dict(q_leg_dict, c=0)
            
            # Replace current_node with q_node in the TTNO
            self.nodes[current_node.identifier] = q_node

            r_node = TensorNode(R, identifier=child_id)

            # We have to add this node as an additional child to the current node
            self.add_child_to_parent(r_node, 0, q_node.identifier, q_node.tensor.ndim - 1)

            if reference_tree.nodes[r_node.identifier].is_leaf():
                new_r_leg_dict = None
            else:
                new_r_leg_dict = TTNO._find_new_leg_dict(r_leg_dict, c=1) # Every r_node will have a parent, so c=1

            self._from_tensor_rec(reference_tree, r_node, new_r_leg_dict)

            # Prepare to repeat for next child
            current_node = self.nodes[current_node.identifier]
            leg_dict = new_q_leg_dict

        return
    
    @staticmethod
    def _prepare_legs_for_QR(reference_tree, current_node, child_id, leg_dict):
        """
        Prepares the lists of legs to be used in the QR-decomposition and
        splits the leg_dict. One of the new dictionary contains the legs
        belonging to the Q-tensor and the other contains the legs belonging to
        the R-tensor

        Parameters
        ----------
        reference_tree : TreeTensorNetwork
            A tree used as a reference. The TTNO will have the same underlying
            tree structure and the same node_ids.
        current_node : TensorNode
            The current node which we want to split via a QR-decomposition.
        child_id : string
            Identifier of the child, which we want to obtain for the TTNO by#
            splitting it of the current_node via a QR-decomposition.
        leg_dict : dict
            A dictionary it contains node_identifiers as keys and leg indices
            as values. It is used to match the legs of tensor in the current
            node to the different nodes it will be split into.

        Returns
        -------
        q_legs: list of int
            The legs of the current_tensor which are to be associated to the
            Q-tensor. (The ones that remain with the current_node)
        r_legs: list of int
            The legs of the current_tensor which are to be associated to the
            R-tensor. (The legs of the to-be created child)
        q_leg_dict, r_leg_dict: dict
            Dictionary describing to which node the open legs of the tensor in
            the Q(R)-node will belong to.
        """
        
        subtree_list_child = reference_tree.find_subtree_of_node(child_id)

        q_legs = []
        r_legs = []
        # Required to keep the leg ordering later as open_legs_in, open_legs_out
        q_legs_sec = []
        r_legs_sec = []

        r_leg_dict = {}
        q_leg_dict = {}


        for node_id_temp in leg_dict:
            if node_id_temp in subtree_list_child:
                r_legs.append(leg_dict[node_id_temp][0])
                r_legs_sec.append(leg_dict[node_id_temp][1])
                r_leg_dict[node_id_temp] = leg_dict[node_id_temp]
            else:
                q_legs.append(leg_dict[node_id_temp][0])
                q_legs_sec.append(leg_dict[node_id_temp][1])
                q_leg_dict[node_id_temp] = leg_dict[node_id_temp]

        q_legs.extend(q_legs_sec)
        r_legs.extend(r_legs_sec)

        if not current_node.is_root():
            q_legs.append(current_node.parent_leg[1])
        q_legs.extend(list(current_node.children_legs.values()))
        
        return q_legs, r_legs, q_leg_dict, r_leg_dict
    
    @staticmethod
    def _create_q_node(tensor, current_node, q_leg_dict):
        """
        Prepares the new TensorNode which will consist of the Q of the 
        QR-decomposition and be in the place of the current_node.

        Parameters
        ----------
        tensor : ndarray
            Tensor associated to the new node.
        current_node : TensorNode
            The node that is currently in the process of being split into two.
        q_leg_dict : dict
            Dictionary describing to which node the open legs of the tensor in
            the new node will belong to.

        Returns
        -------
        q_node : TensorNode
            The new tensor node that will replace the current_node.

        """
        
        q_node = TensorNode(tensor, identifier=current_node.identifier)
        q_half_leg = len(q_leg_dict)
        if not current_node.is_root():
            q_node.open_leg_to_parent(2*q_half_leg, current_node.parent_leg[0])
            c = 1
        else:
            c = 0
            
        q_node.open_legs_to_children(list(range(2*q_half_leg+c, 2*q_half_leg+c + len(current_node.children_legs))),
                                     list(current_node.children_legs.keys()))
        
        return q_node
        
    @staticmethod
    def _find_new_leg_dict(leg_dict, c=0):
        """
        Used to find the Dictionary corresponding to the new tensor created
        in the R-part of the QR-decomposition

        Parameters
        ----------
        leg_dict : dict
            Subdictionary of leg_dict. Keys are node_ids and the values are
            leg indices of open legs of the tensor in the current node, before
            QR-decomposition

        Returns
        -------
        new_leg_dict: dict
            Keys are node_ids and the values are leg indices of open legs in
            the newly obtained tensor node.

        """


        half_leg = len(leg_dict)
        new_leg_dict = {}

        count = 0
        while len(leg_dict) > 0:
            key_w_min_index = ""
            min_index = float("inf")

            for node_id in leg_dict:
                if leg_dict[node_id][0] < min_index:
                    key_w_min_index = node_id
                    min_index = leg_dict[node_id][0]

            # Every node will have two open legs belonging to the
            # same other node are exactly half_leg apart.
            new_leg_dict[key_w_min_index] = [c + count, c + count + half_leg]

            count += 1

            del leg_dict[key_w_min_index]

        return new_leg_dict






