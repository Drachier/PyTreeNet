import unittest
from copy import deepcopy
from typing import Tuple, List

import numpy as np
from scipy.linalg import expm

import pytreenet as ptn
from pytreenet.contractions.state_operator_contraction import (contract_leaf, 
                                                               contract_subtrees_using_dictionary)
from pytreenet.random import (random_tensor_node,
                              random_hermitian_matrix)
from pytreenet.contractions.sandwich_caching import SandwichCache

class TestTDVPonMPS(unittest.TestCase):
    """
    We want to completely test a run of the TDVP algorithm on a simple MPS.
    """
    def _create_mps(self) -> Tuple[ptn.MatrixProductState, ptn.MatrixProductState]:
        """
        Create a simple MPS.
        """
        _, tensor0 = random_tensor_node((4,2), "site_0")
        _, tensor1 = random_tensor_node((4,5,3), "site_1")
        _, tensor2 = random_tensor_node((5,6,4), "site_2")
        _, tensor3 = random_tensor_node((6,5), "site_3")
        tensor_list = [tensor0, tensor1, tensor2, tensor3]
        mps = ptn.MatrixProductState.from_tensor_list(tensor_list,root_site=1,
                                                      node_prefix="site_")
        ref_mps = deepcopy(mps)
        return mps, ref_mps

    def _create_hamiltonian_mpo(self, mps) -> ptn.TTNO:
        """
        Create a simple Hamiltonian as a TTNO.
        """
        matrix = random_hermitian_matrix((2*3*4*5))
        matrix = matrix.reshape(2,3,4,5,2,3,4,5)
        leg_dict = {"site_"+str(i): i for i in range(4)}
        mpo = ptn.TTNO.from_tensor(mps, matrix, leg_dict)
        return mpo

    def _create_operators(self) -> List[ptn.TensorProduct]:
        """
        Generate the operators to be measured.
        """
        operators = [ptn.bosonic_operators(i)[2] for i in range(2,6)]
        operators = [{f"site_{i}": op} for i, op in enumerate(operators)]
        operators = [ptn.TensorProduct(ops) for ops in operators]
        operators.append(ptn.TensorProduct({f"site_{i-2}": np.eye(i) for i in range(2,6)}))
        return operators

    def _init_ref_cache(self, ref_mps, mpo) -> SandwichCache:
        """
        Generate a reference of the inital cache for the TDVP algorithm.
        """
        ref_cache_dict = SandwichCache(ref_mps, mpo)
        node_id = "site_0"
        state_node, state_tensor = ref_mps[node_id]
        op_node, op_tensor = mpo[node_id]
        cache0 = contract_leaf(state_node, state_tensor,
                               op_node, op_tensor)
        ref_cache_dict.add_entry("site_0", "site_1", cache0)
        node_id = "site_1"
        state_node, state_tensor = ref_mps[node_id]
        op_node, op_tensor = mpo[node_id]
        cache1 = contract_subtrees_using_dictionary("site_2",
                                                        state_node, state_tensor,
                                                        op_node, op_tensor,
                                                        ref_cache_dict)
        ref_cache_dict.add_entry("site_1", "site_2", cache1)
        node_id = "site_2"
        state_node, state_tensor = ref_mps[node_id]
        op_node, op_tensor = mpo[node_id]
        cache2 = contract_subtrees_using_dictionary("site_3",
                                                        state_node, state_tensor,
                                                        op_node, op_tensor,
                                                        ref_cache_dict)
        ref_cache_dict.add_entry("site_2", "site_3", cache2)
        return ref_cache_dict

    def _check_cache_initialization(self, tdvp: ptn.FirstOrderOneSiteTDVP,
                                    ref_cache_dict: SandwichCache):
        """
        Check that the cache is correctly initialized.
        """
        self.assertEqual(len(tdvp.partial_tree_cache), 3)
        self.assertTrue(np.allclose(ref_cache_dict.get_entry("site_0", "site_1"),
                                    tdvp.partial_tree_cache.get_entry("site_0", "site_1")))

        self.assertTrue(np.allclose(ref_cache_dict.get_entry("site_1", "site_2"),
                                    tdvp.partial_tree_cache.get_entry("site_1", "site_2")))

        self.assertTrue(np.allclose(ref_cache_dict.get_entry("site_2", "site_3"),
                                    tdvp.partial_tree_cache.get_entry("site_2", "site_3")))

    def _check_init_tdvp1(self, tdvp: ptn.TDVPAlgorithm,
                          ref_mps: ptn.TreeTensorNetworkState):
        """
        Checks that the tdvp algorithm is correctly initialized.
        """
        mps: ptn.TreeTensorNetworkState = tdvp.state
        correct_update_path = ["site_" + str(3-i) for i in range(4)]
        self.assertEqual(correct_update_path, tdvp.update_path)
        correct_orth_path = [[i] for i in correct_update_path[1:]]
        self.assertEqual(correct_orth_path, tdvp.orthogonalization_path)
        self.assertEqual(mps.orthogonality_center_id, "site_3")
        self.assertTrue(mps.is_in_canonical_form("site_3"))
        self.assertEqual(mps,ref_mps)
        ref_cache_dict = self._init_ref_cache(ref_mps, tdvp.hamiltonian)
        self._check_cache_initialization(tdvp, ref_cache_dict)

    def reference_update_of_site_3(self,
                                   ref_mps: ptn.TreeTensorNetworkState,
                                   mpo: ptn.TTNO,
                                   ref_cache_dict: SandwichCache,
                                   time_step_size: float) -> ptn.TreeTensorNetworkState:
        """
        Explicitely computes the update of the site_3 tensor of the MPS.
        """
        heff = np.tensordot(mpo.tensors["site_3"],
                            ref_cache_dict.get_entry("site_2", "site_3"),
                            axes=(mpo.nodes["site_3"].neighbour_index("site_2"),1))
        heff = np.transpose(heff, (3,0,2,1)).reshape(30,30)
        u = expm(-1j*time_step_size*heff)
        u = u.reshape(6,5,6,5)
        ref_mps.tensors["site_3"] = np.tensordot(u, ref_mps.tensors["site_3"],
                                                 axes=((2,3),(0,1)))
        q, r = ptn.tensor_qr_decomposition(ref_mps.tensors["site_3"],
                                           (len(ref_mps.nodes["site_3"].shape)-1, ),
                                           (ref_mps.nodes["site_3"].neighbour_index("site_2"), ),
                                           mode= ptn.SplitMode.KEEP)
        ref_mps.tensors["site_3"] = q.transpose(1,0)
        node_id = "site_3"
        state_node, state_tensor = ref_mps[node_id]
        op_node, op_tensor = mpo[node_id]
        new_cache = contract_leaf(state_node, state_tensor,
                               op_node, op_tensor)
        ref_cache_dict.add_entry("site_3", "site_2", new_cache)
        heff = np.tensordot(ref_cache_dict.get_entry("site_2", "site_3"),
                            ref_cache_dict.get_entry("site_3", "site_2"),
                            axes=(1,1))
        heff = np.transpose(heff, (1,3,0,2)).reshape(36,36)
        u = expm(1j*time_step_size*heff)
        u = u.reshape(6,6,6,6)
        updated_r = np.tensordot(u,r,axes=((2,3),(1,0)))
        next_site = np.tensordot(ref_mps.tensors["site_2"],
                                 updated_r,
                                 axes=(1,0))
        next_site = next_site.transpose(0,2,1)
        ref_mps.tensors["site_2"] = next_site
        ref_mps.orthogonality_center_id = "site_2"
        self.assertTrue(ref_mps.is_in_canonical_form("site_2"))
        return ref_mps

    def reference_update_of_site_2(self,
                                   ref_mps: ptn.TreeTensorNetworkState,
                                   mpo: ptn.TTNO,
                                   ref_cache_dict: SandwichCache,
                                   time_step_size: float) -> ptn.TreeTensorNetworkState:
        """
        Explicitely computes the update of the site_2 tensor of the MPS.
        """
        heff = np.tensordot(ref_cache_dict.get_entry("site_3", "site_2"),
                            mpo.tensors["site_2"],
                            axes=(1,1))
        heff = np.tensordot(ref_cache_dict.get_entry("site_1", "site_2"),
                            heff,
                            axes=(1,2))
        heff = heff.transpose(1,3,4,0,2,5).reshape(120,120)
        u = expm(-1j*time_step_size*heff)
        u = u.reshape(5,6,4,5,6,4)
        updated_site = np.tensordot(u, ref_mps.tensors["site_2"],
                                    axes=((3,4,5),(0,1,2)))
        q, r = ptn.tensor_qr_decomposition(updated_site, (1,2), (0, ),
                                           mode= ptn.SplitMode.KEEP)
        ref_mps.tensors["site_2"] = q.transpose(2,0,1)
        node_id = "site_2"
        state_node, state_tensor = ref_mps[node_id]
        op_node, op_tensor = mpo[node_id]
        new_cache = contract_subtrees_using_dictionary("site_1",
                                                        state_node, state_tensor,
                                                        op_node, op_tensor,
                                                        ref_cache_dict)
        ref_cache_dict.add_entry("site_2", "site_1", new_cache)
        heff = np.tensordot(ref_cache_dict.get_entry("site_1", "site_2"),
                            ref_cache_dict.get_entry("site_2", "site_1"),
                            axes=(1,1))
        heff = np.transpose(heff, (1,3,0,2)).reshape(25,25)
        u = expm(1j*time_step_size*heff)
        u = u.reshape(5,5,5,5)
        updated_r = np.tensordot(u,r,axes=((2,3),(1,0)))
        next_site = np.tensordot(updated_r,
                                 ref_mps.tensors["site_1"],
                                 axes=(0,0))
        ref_mps.tensors["site_1"] = next_site
        ref_mps.orthogonality_center_id = "site_1"
        self.assertTrue(ref_mps.is_in_canonical_form("site_1"))
        return ref_mps

    def reference_update_of_site_1(self,
                                   ref_mps: ptn.TreeTensorNetworkState,
                                   mpo: ptn.TTNO,
                                   ref_cache_dict: SandwichCache,
                                   time_step_size: float) -> ptn.TreeTensorNetworkState:
        """
        Explicitely computes the update of the site_1 tensor of the MPS.
        """
        heff = np.tensordot(mpo.tensors["site_1"],
                            ref_cache_dict.get_entry("site_2", "site_1"),
                            axes=(1,1))
        heff = np.tensordot(heff,
                            ref_cache_dict.get_entry("site_0", "site_1"),
                            axes=(0,1))
        heff = heff.transpose(3,5,0,2,4,1)
        heff = heff.reshape(60,60)
        u = expm(-1j*time_step_size*heff)
        u = u.reshape(5,4,3,5,4,3)
        updated_site = np.tensordot(u, ref_mps.tensors["site_1"],
                                    axes=((3,4,5),(0,1,2)))
        q, r = ptn.tensor_qr_decomposition(updated_site, (0,2), (1, ),
                                             mode= ptn.SplitMode.KEEP)
        ref_mps.tensors["site_1"] = q.transpose(0,2,1)
        # In the actual TDVP the order of children changed
        ref_mps.nodes["site_1"].swap_two_child_legs("site_0","site_2")
        node_id = "site_1"
        state_node, state_tensor = ref_mps[node_id]
        op_node, op_tensor = mpo[node_id]
        new_cache = contract_subtrees_using_dictionary("site_0",
                                                        state_node, state_tensor,
                                                        op_node, op_tensor,
                                                        ref_cache_dict)
        ref_cache_dict.add_entry("site_1", "site_0", new_cache)
        heff = np.tensordot(ref_cache_dict.get_entry("site_1", "site_0"),
                            ref_cache_dict.get_entry("site_0", "site_1"),
                            axes=(1,1))
        heff = np.transpose(heff, (1,3,0,2)).reshape(16,16)
        u = expm(1j*time_step_size*heff)
        u = u.reshape(4,4,4,4)
        updated_r = np.tensordot(u,r,axes=((2,3),(0,1)))
        next_site = np.tensordot(updated_r,
                                 ref_mps.tensors["site_0"],
                                 axes=(1,0))
        ref_mps.tensors["site_0"] = next_site
        ref_mps.orthogonality_center_id = "site_0"
        self.assertTrue(ref_mps.is_in_canonical_form("site_0"))
        ref_cache_dict = self._init_ref_cache(ref_mps, mpo)
        return ref_mps

    def reference_update_of_site_0(self,
                                   ref_mps: ptn.TreeTensorNetworkState,
                                   mpo: ptn.TTNO,
                                   ref_cache_dict: SandwichCache,
                                   time_step_size: float) -> Tuple[ptn.TreeTensorNetworkState,SandwichCache]:
        """
        Explicitely computes the update of the site_0 tensor of the MPS.
        """
        heff = np.tensordot(mpo.tensors["site_0"],
                            ref_cache_dict.get_entry("site_1", "site_0"),
                            axes=(0,1))
        heff = heff.transpose(3,0,2,1)
        heff = heff.reshape(8,8)
        u = expm(-1j*time_step_size*heff)
        u = u.reshape(4,2,4,2)
        ref_mps.tensors["site_0"] = np.tensordot(u, ref_mps.tensors["site_0"],
                                                 axes=((2,3),(0,1)))
        ref_mps.move_orthogonalization_center("site_3",
                                              mode=ptn.SplitMode.KEEP)
        ref_cache_dict = self._init_ref_cache(ref_mps, mpo)
        return ref_mps, ref_cache_dict

    def test_main(self):
        # Preparing the tensor structures
        mps, ref_mps = self._create_mps()
        self.assertEqual(mps,ref_mps)
        mpo = self._create_hamiltonian_mpo(mps)
        self.assertEqual(set(mpo.nodes.keys()),set(mps.nodes.keys()))
        operators = self._create_operators()
        self.assertEqual(len(operators)-1,len(mps.nodes))

        # Generating the TDVP algorithm
        time_step_size = 0.1
        final_time = 1.0
        tdvp = ptn.FirstOrderOneSiteTDVP(mps, mpo,
                                         time_step_size, final_time,
                                         operators)
        mps : ptn.TreeTensorNetworkState = tdvp.state
        ref_mps.orthogonalize("site_3", mode=ptn.SplitMode.KEEP)

        # Checking for correct initialization
        self._check_init_tdvp1(tdvp, ref_mps)
        ref_cache_dict = self._init_ref_cache(ref_mps, mpo)

        # Running the first time_step
        ## Updating Site 3
        ref_mps = self.reference_update_of_site_3(ref_mps, mpo,
                                                  ref_cache_dict,
                                                  time_step_size)
        tdvp._first_update("site_3")
        self.assertEqual("site_2", mps.orthogonality_center_id)
        self.assertTrue(mps.is_in_canonical_form("site_2"))
        self.assertEqual(ref_mps,mps)
        self.assertTrue(np.allclose(ref_cache_dict.get_entry("site_3", "site_2"),
                                    tdvp.partial_tree_cache.get_entry("site_3", "site_2")))
        self.assertTrue(np.allclose(ref_cache_dict.get_entry("site_2", "site_3"),
                                    tdvp.partial_tree_cache.get_entry("site_2", "site_3")))

        ## Updating Site 2
        ref_mps = self.reference_update_of_site_2(ref_mps, mpo,
                                                  ref_cache_dict,
                                                  time_step_size)
        tdvp._normal_update("site_2", 1)
        self.assertEqual("site_1", mps.orthogonality_center_id)
        self.assertTrue(mps.is_in_canonical_form("site_1"))
        self.assertEqual(ref_mps,mps)
        self.assertTrue(np.allclose(ref_cache_dict.get_entry("site_2", "site_1"),
                                    tdvp.partial_tree_cache.get_entry("site_2", "site_1")))
        self.assertTrue(np.allclose(ref_cache_dict.get_entry("site_1", "site_2"),
                                    tdvp.partial_tree_cache.get_entry("site_1", "site_2")))

        ## Updating Site 1
        ref_mps = self.reference_update_of_site_1(ref_mps, mpo,
                                                  ref_cache_dict,
                                                  time_step_size)
        tdvp._normal_update("site_1", 2)
        self.assertEqual("site_0", mps.orthogonality_center_id)
        self.assertTrue(mps.is_in_canonical_form("site_0"))
        self.assertTrue(np.allclose(ref_cache_dict.get_entry("site_1", "site_0"),
                                    tdvp.partial_tree_cache.get_entry("site_1", "site_0")))
        self.assertTrue(np.allclose(ref_cache_dict.get_entry("site_0", "site_1"),
                                    tdvp.partial_tree_cache.get_entry("site_0", "site_1")))

        self.assertEqual(ref_mps,mps)

        ## Updating Site 0
        ref_mps, ref_cache_dict = self.reference_update_of_site_0(ref_mps, mpo,
                                                                  ref_cache_dict,
                                                                  time_step_size)
        tdvp._final_update("site_0")
        self.assertEqual("site_3", mps.orthogonality_center_id)
        self.assertTrue(mps.is_in_canonical_form("site_3"))
        self.assertEqual(ref_mps,mps)
        pairs_to_check = [("site_0", "site_1"), ("site_1", "site_2"), ("site_2", "site_3")]
        for node_id, next_node_id in pairs_to_check:
            self.assertTrue(np.allclose(ref_cache_dict.get_entry(node_id, next_node_id),
                                        tdvp.partial_tree_cache.get_entry(node_id, next_node_id)))
        # Thus all functions running one after the other work as expected
        # Now we want to test the run_one_time_step function

    def test_run_one_time_step(self):
        mps, _ = self._create_mps()
        mpo = self._create_hamiltonian_mpo(mps)
        operators = self._create_operators()
        time_step_size = 0.1
        final_time = 1.0
        tdvp = ptn.FirstOrderOneSiteTDVP(mps, mpo,
                                         time_step_size, final_time,
                                         operators)
        ref_tdvp = deepcopy(tdvp)

        # Reference Evolution
        ref_tdvp._first_update("site_3")
        ref_tdvp._normal_update("site_2", 1)
        ref_tdvp._normal_update("site_1", 2)
        ref_tdvp._final_update("site_0")
        # Actual Evolution
        tdvp.run_one_time_step()

        self.assertEqual(ref_tdvp.state, tdvp.state)
        pairs_to_check = [("site_0", "site_1"), ("site_1", "site_2"), ("site_2", "site_3")]
        for node_id, next_node_id in pairs_to_check:
            self.assertTrue(np.allclose(ref_tdvp.partial_tree_cache.get_entry(node_id, next_node_id),
                                        tdvp.partial_tree_cache.get_entry(node_id, next_node_id)))

    def _check_init_tdvp2(self, tdvp: ptn.SecondOrderOneSiteTDVP):
        """
        Performs the additional checks needed for the second order TDVP.
        """
        correct_backwards_up_path = ["site_" + str(i) for i in range(4)]
        self.assertEqual(correct_backwards_up_path, tdvp.backwards_update_path)
        correct_backwards_orth_path = [[i] for i in correct_backwards_up_path[1:]]
        self.assertEqual(correct_backwards_orth_path, tdvp.backwards_orth_path)

    def _reference_final_forward_update(self,
                                        ref_tdvp: ptn.FirstOrderOneSiteTDVP) -> ptn.FirstOrderOneSiteTDVP:
        """
        A reference implmentation of the final forward update.
        The final node must be time evolved by 2*(time_step_size/2),
         i.e. with a normal time-step.
        """
        ref_mps = ref_tdvp.state
        mpo = ref_tdvp.hamiltonian
        cache_dict = ref_tdvp.partial_tree_cache
        heff = np.tensordot(mpo.tensors["site_0"],
                            cache_dict.get_entry("site_1", "site_0"),
                            axes=(0,1))
        heff = heff.transpose(3,0,2,1)
        heff = heff.reshape(8,8)
        time_step_size = 2 * ref_tdvp.time_step_size
        u = expm(-1j*time_step_size*heff)
        u = u.reshape(4,2,4,2)
        ref_mps.tensors["site_0"] = np.tensordot(u,
                                                 ref_mps.tensors["site_0"],
                                                 axes=((2,3),(0,1)))
        self.assertEqual("site_0", ref_mps.orthogonality_center_id)
        return ref_tdvp

    def _reference_first_backwards_link_update(self,
                                               ref_tdvp: ptn.FirstOrderOneSiteTDVP) -> ptn.FirstOrderOneSiteTDVP:
        """
        A reference implmentation of the first backwards link update.
         In this case the link between site_0 and site_1 is updated.
        """
        ref_mps = ref_tdvp.state
        mpo = ref_tdvp.hamiltonian
        cache_dict = ref_tdvp.partial_tree_cache
        q, r = ptn.tensor_qr_decomposition(ref_mps.tensors["site_0"],
                                           (1, ), (0, ),
                                           mode=ptn.SplitMode.KEEP)
        ref_mps.tensors["site_0"] = q.transpose(1,0)
        node_id = "site_0"
        state_node, state_tensor = ref_mps[node_id]
        op_node, op_tensor = mpo[node_id]
        new_cache = contract_leaf(state_node, state_tensor,
                               op_node, op_tensor)
        cache_dict.add_entry("site_0", "site_1", new_cache)
        heff = np.tensordot(cache_dict.get_entry("site_0", "site_1"),
                            cache_dict.get_entry("site_1", "site_0"),
                            axes=(1,1))
        heff = np.transpose(heff, (1,3,0,2)).reshape(16,16)
        time_step_size = ref_tdvp.time_step_size
        u = expm(1j*time_step_size*heff)
        u = u.reshape(4,4,4,4)
        updated_r = np.tensordot(u,r,axes=((2,3),(0,1)))
        next_site = np.tensordot(updated_r,
                                 ref_mps.tensors["site_1"],
                                 axes=(1,0))
        ref_mps.tensors["site_1"] = next_site
        ref_mps.orthogonality_center_id = "site_1"
        return ref_tdvp

    def _reference_backward_update_site_1(self,
                                          ref_tdvp: ptn.FirstOrderOneSiteTDVP) -> ptn.FirstOrderOneSiteTDVP:
        """
        A reference implmentation of the backward update of the site_1 tensor.
        """
        ref_mps = ref_tdvp.state
        mpo = ref_tdvp.hamiltonian
        cache_dict = ref_tdvp.partial_tree_cache
        # Site update
        heff = np.tensordot(mpo.tensors["site_1"],
                            cache_dict.get_entry("site_0", "site_1"),
                            axes=(0,1))
        heff = np.tensordot(heff,
                            cache_dict.get_entry("site_2", "site_1"),
                            axes=(0,1))
        heff = heff.transpose(3,5,0,2,4,1)
        heff = heff.reshape(60,60)
        time_step_size = ref_tdvp.time_step_size
        u = expm(-1j*time_step_size*heff)
        u = u.reshape(4,5,3,4,5,3)
        next_site = ref_mps.tensors["site_1"]
        updated_site = np.tensordot(u, next_site,
                                    axes=((3,4,5),(0,1,2)))
        ref_mps.tensors["site_1"] = updated_site
        self.assertTrue(ref_mps.is_in_canonical_form("site_1"))
        q, r = ptn.tensor_qr_decomposition(ref_mps.tensors["site_1"],
                                           (0,2), (1, ),
                                           mode= ptn.SplitMode.KEEP)
        ref_mps.tensors["site_1"] = q.transpose(0,2,1)
        # The order of the two children is changed in the actual tdvp
        ref_mps.nodes["site_1"].swap_two_child_legs("site_0","site_2")
        node_id = "site_1"
        state_node, state_tensor = ref_mps[node_id]
        op_node, op_tensor = mpo[node_id]
        new_cache = contract_subtrees_using_dictionary("site_2",
                                                        state_node, state_tensor,
                                                        op_node, op_tensor,
                                                        cache_dict)
        cache_dict.add_entry("site_1", "site_2", new_cache)
        # Link update
        heff = np.tensordot(cache_dict.get_entry("site_1", "site_2"),
                            cache_dict.get_entry("site_2", "site_1"),
                            axes=(1,1))
        heff = np.transpose(heff, (1,3,0,2)).reshape(25,25)
        u = expm(1j*time_step_size*heff)
        u = u.reshape(5,5,5,5)
        updated_r = np.tensordot(u,r,axes=((2,3),(0,1)))
        next_site = np.tensordot(updated_r,
                                 ref_mps.tensors["site_2"],
                                 axes=(1,0))
        ref_mps.tensors["site_2"] = next_site
        ref_mps.orthogonality_center_id = "site_2"
        self.assertTrue(ref_mps.is_in_canonical_form("site_2"))
        return ref_tdvp
    
    def _reference_backward_update_site_2(self,
                                          ref_tdvp: ptn.FirstOrderOneSiteTDVP) -> ptn.FirstOrderOneSiteTDVP:
        """
        A reference implmentation of the backward update of the site_2 tensor.
        """
        ref_mps = ref_tdvp.state
        mpo = ref_tdvp.hamiltonian
        cache_dict = ref_tdvp.partial_tree_cache
        # Site update
        heff = np.tensordot(mpo.tensors["site_2"],
                            cache_dict.get_entry("site_1", "site_2"),
                            axes=(0,1))
        heff = np.tensordot(heff,
                            cache_dict.get_entry("site_3", "site_2"),
                            axes=(0,1))
        heff = heff.transpose(3,5,0,2,4,1)
        heff = heff.reshape(120,120)
        time_step_size = ref_tdvp.time_step_size
        u = expm(-1j*time_step_size*heff)
        u = u.reshape(5,6,4,5,6,4)
        next_site = ref_mps.tensors["site_2"]
        updated_site = np.tensordot(u, next_site,
                                    axes=((3,4,5),(0,1,2)))
        ref_mps.tensors["site_2"] = updated_site
        self.assertTrue(ref_mps.is_in_canonical_form("site_2"))
        # Link update
        q, r = ptn.tensor_qr_decomposition(ref_mps.tensors["site_2"],
                                           (0,2), (1, ),
                                           mode= ptn.SplitMode.KEEP)
        ref_mps.tensors["site_2"] = q.transpose(0,2,1)
        node_id = "site_2"
        state_node, state_tensor = ref_mps[node_id]
        op_node, op_tensor = mpo[node_id]
        new_cache = contract_subtrees_using_dictionary("site_3",
                                                        state_node, state_tensor,
                                                        op_node, op_tensor,
                                                        cache_dict)
        cache_dict.add_entry("site_2", "site_3", new_cache)
        heff = np.tensordot(cache_dict.get_entry("site_2", "site_3"),
                            cache_dict.get_entry("site_3", "site_2"),
                            axes=(1,1))
        heff = np.transpose(heff, (1,3,0,2)).reshape(36,36)
        u = expm(1j*time_step_size*heff)
        u = u.reshape(6,6,6,6)
        updated_r = np.tensordot(u,r,axes=((2,3),(0,1)))
        next_site = np.tensordot(updated_r,
                                 ref_mps.tensors["site_3"],
                                 axes=(1,0))
        ref_mps.tensors["site_3"] = next_site
        ref_mps.orthogonality_center_id = "site_3"
        self.assertTrue(ref_mps.is_in_canonical_form("site_3"))
        return ref_tdvp
    
    def _reference_final_backwards_update(self,
                                          ref_tdvp: ptn.FirstOrderOneSiteTDVP) -> ptn.FirstOrderOneSiteTDVP:
        """
        A reference implementation of the final backwards update.
        """
        ref_mps = ref_tdvp.state
        mpo = ref_tdvp.hamiltonian
        cache_dict = ref_tdvp.partial_tree_cache
        # Site update
        heff = np.tensordot(mpo.tensors["site_3"],
                            cache_dict.get_entry("site_2", "site_3"),
                            axes=(0,1))
        heff = heff.transpose(3,0,2,1)
        heff = heff.reshape(30,30)
        time_step_size =  ref_tdvp.time_step_size
        u = expm(-1j*time_step_size*heff)
        u = u.reshape(6,5,6,5)
        next_site = ref_mps.tensors["site_3"]
        updated_site = np.tensordot(u, next_site,
                                    axes=((2,3),(0,1)))
        ref_mps.tensors["site_3"] = updated_site
        self.assertTrue(ref_mps.is_in_canonical_form("site_3"))
        return ref_tdvp

    def test_second_order_tdvp_main(self):
        """
        This function tests a full time step, i.e. a left to right sweep of
         the second order TDVP algorithm on an MPS against a manual reference
         computation.
        """
        # Preparing the tensor structures
        mps, _ = self._create_mps()
        mpo = self._create_hamiltonian_mpo(mps)
        operators = self._create_operators()

        # Initialise the TDVP algorithms
        time_step_size = 0.1
        final_time = 1.0
        tdvp = ptn.SecondOrderOneSiteTDVP(mps, mpo,
                                          time_step_size, final_time,
                                          operators)
        ref_tdvp =  ptn.FirstOrderOneSiteTDVP(mps, mpo,
                                              time_step_size / 2,
                                              final_time,
                                              operators)
        mps : ptn.TreeTensorNetworkState = tdvp.state
        ref_mps : ptn.TreeTensorNetworkState = ref_tdvp.state

        # Checking for correct initialization
        self._check_init_tdvp1(tdvp, ref_mps)
        self._check_init_tdvp2(tdvp)

        # Running forward sweep
        ref_tdvp._first_update("site_3")
        ref_tdvp._normal_update("site_2", 1)
        ref_tdvp._normal_update("site_1", 2)
        tdvp.forward_sweep()
        self.assertEqual(ref_tdvp.state, tdvp.state)
        pairs_to_check = [("site_3", "site_2"), ("site_2", "site_1"), ("site_1", "site_0")]
        for node_id, next_node_id in pairs_to_check:
            self.assertTrue(np.allclose(ref_tdvp.partial_tree_cache.get_entry(node_id, next_node_id),
                                        tdvp.partial_tree_cache.get_entry(node_id, next_node_id)))

        ## Running the final forward update
        ref_tdvp = self._reference_final_forward_update(ref_tdvp)
        tdvp._final_forward_update()
        self.assertEqual(ref_mps, mps)
        self.assertEqual(ref_mps.orthogonality_center_id,
                         mps.orthogonality_center_id)

        # Running the backward sweep
        ## First backward update of a link
        ref_tdvp = self._reference_first_backwards_link_update(ref_tdvp)
        tdvp._update_first_backward_link()
        self.assertEqual(ref_mps, mps)
        self.assertTrue(np.allclose(ref_tdvp.partial_tree_cache.get_entry("site_0", "site_1"),
                                    tdvp.partial_tree_cache.get_entry("site_0", "site_1")))

        ## Running the backward update of site_1
        ref_tdvp = self._reference_backward_update_site_1(ref_tdvp)
        tdvp._normal_backward_update("site_1", 1)
        self.assertEqual(ref_mps, mps)
        self.assertTrue(np.allclose(ref_tdvp.partial_tree_cache.get_entry("site_1", "site_2"),
                                    tdvp.partial_tree_cache.get_entry("site_1", "site_2")))

        ## Running the backward update of site_2
        ref_tdvp = self._reference_backward_update_site_2(ref_tdvp)
        tdvp._normal_backward_update("site_2", 2)
        self.assertEqual(ref_mps, mps)
        self.assertTrue(np.allclose(ref_tdvp.partial_tree_cache.get_entry("site_2", "site_3"),
                                    tdvp.partial_tree_cache.get_entry("site_2", "site_3")))

        ## Running the final backward update
        ref_tdvp = self._reference_final_backwards_update(ref_tdvp)
        tdvp._final_backward_update()
        self.assertEqual(ref_mps, mps)
        self.assertTrue(mps.is_in_canonical_form("site_3"))

    def test_second_order_tdvp_run_one_time_step(self):
        # Preparing the tensor structures
        mps, _ = self._create_mps()
        mpo = self._create_hamiltonian_mpo(mps)
        operators = self._create_operators()

        # Initialise the TDVP algorithms
        time_step_size = 0.1
        final_time = 1.0
        tdvp = ptn.SecondOrderOneSiteTDVP(mps, mpo,
                                          time_step_size, final_time,
                                          operators)
        ref_tdvp =  deepcopy(tdvp)

        # Running reference using implemented functions
        ref_tdvp.forward_sweep()
        ref_tdvp._final_forward_update()
        ref_tdvp._update_first_backward_link()
        ref_tdvp._normal_backward_update("site_1", 1)
        ref_tdvp._normal_backward_update("site_2", 2)
        ref_tdvp._final_backward_update()

        # Running the actual TDVP
        tdvp.run_one_time_step()

        self.assertEqual(ref_tdvp.state, tdvp.state)
        pairs_to_check = [("site_2", "site_3"), ("site_1", "site_2"), ("site_0", "site_1")]
        for node_id, next_node_id in pairs_to_check:
            self.assertTrue(np.allclose(ref_tdvp.partial_tree_cache.get_entry(node_id, next_node_id),
                                        tdvp.partial_tree_cache.get_entry(node_id, next_node_id)))

if __name__ == "__main__":
    unittest.main()
