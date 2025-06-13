import unittest

import numpy as np
from scipy.linalg import expm

import pytreenet as ptn
from pytreenet.random import (random_small_ttns,
                              random_big_ttns_two_root_children,
                              random_hermitian_matrix,
                              RandomTTNSMode)

class TestSWAPList(unittest.TestCase):
    def setUp(self) -> None:
        self.small_ttn = random_small_ttns(mode=RandomTTNSMode.SAMEPHYS)
        self.big_ttn = random_big_ttns_two_root_children()
        self.small_swaps = [("c1","root"),("root","c2")]

        self.ref_swap = np.asarray([[1,0,0,0],
                                    [0,0,1,0],
                                    [0,1,0,0],
                                    [0,0,0,1]])
        self.ref_swap = self.ref_swap.reshape(2,2,2,2)

    def test_init_failure_one(self):
        """
        Ensure the initialisation fails if only one identifier is given.
        """
        self.small_swaps.append(("c1",))
        self.assertRaises(ValueError,ptn.SWAPlist,self.small_swaps)

    def test_init_failure_three(self):
        """
        Ensure the initilisation fails if more than two identifiers are given.
        """
        self.small_swaps.append(("c1","c2","root"))
        self.assertRaises(ValueError,ptn.SWAPlist,self.small_swaps)

    def test_succesfull_init(self):
        """
        Tests a succesfull initilisation.
        """
        swaps = ptn.SWAPlist(self.small_swaps)
        for ids in self.small_swaps:
            self.assertTrue(ids in swaps)
        self.assertEqual(swaps,self.small_swaps)

    def test_empty_init(self):
        """
        Test initialisation without swap specification.
        """
        swaps = ptn.SWAPlist()
        self.assertEqual(0,len(swaps))

    def test_compatability_small_true(self):
        """
        Test the compatability with a small TTNS.
        """
        swaps = ptn.SWAPlist(self.small_swaps)
        self.assertTrue(swaps.is_compatible_with_ttn(self.small_ttn))

    def test_compatability_small_false_non_ex(self):
        """
        Tests the compatability with a small TTNS, if the first SWAP partner is
        not in the TTNS.
        """
        self.small_swaps.append(("not a node","root"))
        swaps = ptn.SWAPlist(self.small_swaps)
        self.assertFalse(swaps.is_compatible_with_ttn(self.small_ttn))

    def test_compatability_small_false_no_neigh(self):
        """
        Tests the compatability with a small TTNS, if the second SWAP partner
        is not a neighbour of the first.
        """
        self.small_swaps.append(("c1","c2"))
        swaps = ptn.SWAPlist(self.small_swaps)
        self.assertFalse(swaps.is_compatible_with_ttn(self.small_ttn))

    def test_compatability_small_false_wrong_dim(self):
        """
        Tests the compatability with a small TTNS, if the two nodes have a
        different open leg dimension.
        """
        ttn = random_small_ttns()
        swaps = ptn.SWAPlist(self.small_swaps)
        self.assertFalse(swaps.is_compatible_with_ttn(ttn))

    def test_into_operator_dim(self):
        """
        Tests the method to turn a SWAPList into operators, if the dimension is
        specified.
        """
        swaps = ptn.SWAPlist(self.small_swaps)
        ops = swaps.into_operators(dim=2)
        for i, op in enumerate(ops):
            self.assertEqual(list(self.small_swaps[i]),op.node_identifiers)
            self.assertTrue(np.allclose(self.ref_swap,op.operator))

    def test_into_operator_ttn(self):
        """
        Tests the method to turn a SWAPList into operators, if the reference
        ttn is specified.
        """
        swaps = ptn.SWAPlist(self.small_swaps)
        ops = swaps.into_operators(ttn=self.small_ttn)
        for i, op in enumerate(ops):
            self.assertEqual(list(self.small_swaps[i]),op.node_identifiers)
            self.assertTrue(np.allclose(self.ref_swap,op.operator))

    def test_into_operator_none(self):
        """
        Tests the method to turn a SWAPList into operators, if neither the
        dimension nor the ttn is specified. An Exception should be thrown.
        """
        swaps = ptn.SWAPlist(self.small_swaps)
        self.assertRaises(ValueError,swaps.into_operators)

class TestTrotterStep(unittest.TestCase):

    def setUp(self):
        self.small_swaps = [("c1","root"),("root","c2")]
        self.small_swaps = ptn.SWAPlist(self.small_swaps)
        self.ref_swap = np.asarray([[1,0,0,0],
                                    [0,0,1,0],
                                    [0,1,0,0],
                                    [0,0,0,1]])
        self.ref_swap = self.ref_swap.reshape(2,2,2,2)
        op1 = random_hermitian_matrix(2)
        op2 = random_hermitian_matrix(2)
        self.operator = ptn.TensorProduct({"c1": op1, "root": op2})
        self.factor = 0.5
        self.trotter = ptn.TrotterStep(self.operator,
                                       self.factor,
                                       self.small_swaps,
                                       self.small_swaps)
        self.trotter_no_swaps = ptn.TrotterStep(self.operator,
                                                self.factor)

    def test_realise_swaps_full(self):
        """
        Test the realisation of swap gates in a trotter steps.
        """
        swap_bef, swap_aft = self.trotter.realise_swaps(dim=2)
        for i, op in enumerate(swap_bef):
            self.assertEqual(list(self.small_swaps[i]),op.node_identifiers)
            self.assertTrue(np.allclose(self.ref_swap,op.operator))
            op2 = swap_aft[i]
            self.assertEqual(list(self.small_swaps[i]),op2.node_identifiers)
            self.assertTrue(np.allclose(self.ref_swap,op2.operator))

    def test_realise_swaps_empty(self):
        """
        Test the realisation of swap gates in a trotter steps.
        """
        swap_bef, swap_aft = self.trotter_no_swaps.realise_swaps(dim=2)
        self.assertEqual(0,len(swap_bef))
        self.assertEqual(0,len(swap_aft))

    def test_exponentiate_operator(self):
        """
        Test the exponentiation of the tensor product.
        """
        delta_t = 0.2
        operator = np.kron(self.operator["c1"],self.operator["root"])
        ref_op = expm(-1j * self.factor * delta_t * operator)
        ref_op = ref_op.reshape(2,2,2,2)
        ref_ids = ["c1","root"]
        found_num_op = self.trotter.exponentiate_operator(delta_t,
                                                          dim=2)
        self.assertTrue(np.allclose(ref_op,found_num_op.operator))
        self.assertEqual(ref_ids,found_num_op.node_identifiers)

    def test_exponentiate_operator_matrix(self):
        """
        Exponentiate an operator that is just a matrix.
        """
        delta_t = 0.2
        operator = random_hermitian_matrix(2)
        ref_exp = expm(-1j * self.factor * delta_t * operator)
        trotter_step = ptn.TrotterStep(ptn.TensorProduct({"c1":operator}),
                                       self.factor)
        found_num_op = trotter_step.exponentiate_operator(delta_t,dim=2)
        self.assertTrue(np.allclose(ref_exp,found_num_op.operator))
        self.assertEqual(["c1"],found_num_op.node_identifiers)

class TestTrotterSplittingInit(unittest.TestCase):

    def setUp(self):
        self.small_swaps = [("c1","root"),("root","c2")]
        self.small_swaps = ptn.SWAPlist(self.small_swaps)
        self.operators = [random_hermitian_matrix(2) for _ in range(4)]
        self.tp = [ptn.TensorProduct({"c1": self.operators[0],
                                      "root": self.operators[1]}),
                   ptn.TensorProduct({"c1": self.operators[0],
                                      "root": self.operators[1]}),
                   ptn.TensorProduct({"root": self.operators[2],
                                      "c2": self.operators[3]})
                   ]
        self.factor1 = 0.5
        self.trotter1 = ptn.TrotterStep(self.tp[0],
                                        self.factor1,
                                        self.small_swaps,
                                        self.small_swaps)
        self.trotter_no_swaps = ptn.TrotterStep(self.tp[1],1)
        self.factor2 = 0.2
        self.trotter2 = ptn.TrotterStep(self.tp[2],
                                        self.factor2,
                                        self.small_swaps,
                                        self.small_swaps)

        self.splitting = [(0,self.factor1),
                          (1,1),
                          (2,self.factor2)]
        self.swaps_before = [self.small_swaps, [], self.small_swaps]
        self.swaps_after = [self.small_swaps, [], self.small_swaps]

    def test_init(self):
        """
        Test a normal initilisation of the Trotterisation.
        """
        trotterisation = ptn.TrotterSplitting([self.trotter1,
                                               self.trotter_no_swaps,
                                               self.trotter2])
        self.assertEqual(3,len(trotterisation))

    def test_init_from_lists_full(self):
        """
        Test an initialisation from lists with all keyword arguments.
        """
        trotterisation = ptn.TrotterSplitting.from_lists(self.tp,
                                                         self.splitting,
                                                         self.swaps_before,
                                                         self.swaps_after)
        for i, trotterstep in enumerate(trotterisation):
            ref_tp = self.tp[i]
            found_tp = trotterstep.operator
            self.assertTrue(ref_tp.allclose(found_tp))
            ref_factor = self.splitting[i][1]
            found_factor = trotterstep.factor
            self.assertEqual(ref_factor,found_factor)
            ref_swapsb = self.swaps_before[i]
            found_swapsb = trotterstep.swaps_after
            self.assertEqual(ref_swapsb,found_swapsb)
            ref_swapsa = self.swaps_after[i]
            found_swapsa = trotterstep.swaps_after
            self.assertEqual(ref_swapsa,found_swapsa)

    def test_init_from_lists_integer_splitting(self):
        """
        Test an initialisation from lists with all keyword arguments, but
        where the splitting is a list of integers.
        """
        splitting = [sp[0] for sp in self.splitting]
        trotterisation = ptn.TrotterSplitting.from_lists(self.tp,
                                                         splitting,
                                                         self.swaps_before,
                                                         self.swaps_after)
        for i, trotterstep in enumerate(trotterisation):
            ref_tp = self.tp[i]
            found_tp = trotterstep.operator
            self.assertTrue(ref_tp.allclose(found_tp))
            ref_factor = 1
            found_factor = trotterstep.factor
            self.assertEqual(ref_factor,found_factor)
            ref_swapsb = self.swaps_before[i]
            found_swapsb = trotterstep.swaps_after
            self.assertEqual(ref_swapsb,found_swapsb)
            ref_swapsa = self.swaps_after[i]
            found_swapsa = trotterstep.swaps_after
            self.assertEqual(ref_swapsa,found_swapsa)

    def test_init_from_lists_no_splitting(self):
        """
        Test an initialisation from lists with no splitting given.
        """
        trotterisation = ptn.TrotterSplitting.from_lists(self.tp,
                                                         swaps_before=self.swaps_before,
                                                         swaps_after=self.swaps_after)
        for i, trotterstep in enumerate(trotterisation):
            ref_tp = self.tp[i]
            found_tp = trotterstep.operator
            self.assertTrue(ref_tp.allclose(found_tp))
            ref_factor = 1
            found_factor = trotterstep.factor
            self.assertEqual(ref_factor,found_factor)
            ref_swapsb = self.swaps_before[i]
            found_swapsb = trotterstep.swaps_after
            self.assertEqual(ref_swapsb,found_swapsb)
            ref_swapsa = self.swaps_after[i]
            found_swapsa = trotterstep.swaps_after
            self.assertEqual(ref_swapsa,found_swapsa)

    def test_init_from_lists_no_swaps(self):
        """
        Test an initialisation from lists with no swap lists given.
        """
        trotterisation = ptn.TrotterSplitting.from_lists(self.tp)
        for i, trotterstep in enumerate(trotterisation):
            ref_tp = self.tp[i]
            found_tp = trotterstep.operator
            self.assertTrue(ref_tp.allclose(found_tp))
            ref_factor = 1
            found_factor = trotterstep.factor
            self.assertEqual(ref_factor,found_factor)
            ref_swapsb = ptn.SWAPlist()
            found_swapsb = trotterstep.swaps_after
            self.assertEqual(ref_swapsb,found_swapsb)
            ref_swapsa = ptn.SWAPlist()
            found_swapsa = trotterstep.swaps_after
            self.assertEqual(ref_swapsa,found_swapsa)

class TestTrotterSplittingMethods(unittest.TestCase):

    def setUp(self):
        self.small_swaps = [("c1","root"),("root","c2")]
        self.small_swaps = ptn.SWAPlist(self.small_swaps)
        self.ref_swap = np.asarray([[1,0,0,0],
                                    [0,0,1,0],
                                    [0,1,0,0],
                                    [0,0,0,1]])
        self.ref_swap = self.ref_swap.reshape(2,2,2,2)
        self.operators = [random_hermitian_matrix(2) for _ in range(4)]
        self.tp = [ptn.TensorProduct({"c1": self.operators[0],
                                      "root": self.operators[1]}),
                   ptn.TensorProduct({"c1": self.operators[0],
                                      "root": self.operators[1]}),
                   ptn.TensorProduct({"root": self.operators[2],
                                      "c2": self.operators[3]})
                   ]
        self.factor1 = 0.5
        self.trotter1 = ptn.TrotterStep(self.tp[0],
                                        self.factor1,
                                        self.small_swaps,
                                        self.small_swaps)
        self.trotter_no_swaps = ptn.TrotterStep(self.tp[1],1)
        self.factor2 = 0.2
        self.trotter2 = ptn.TrotterStep(self.tp[2],
                                        self.factor2,
                                        self.small_swaps,
                                        self.small_swaps)
        self.trottersplit = ptn.TrotterSplitting([self.trotter1,
                                                  self.trotter_no_swaps,
                                                  self.trotter2])

    def test_exponentiation(self):
        """
        Test the exponentiation of an entire trotter splitting.
        """
        delta_t = 0.1
        dim = 2
        small_swaps_list = [list(swap) for swap in self.small_swaps]
        found_unitaries = self.trottersplit.exponentiate_splitting(delta_t,
                                                                   dim=dim)
        # Step 1
        self.assertTrue(np.allclose(self.ref_swap,found_unitaries[0].operator))
        self.assertEqual(small_swaps_list[0],
                         found_unitaries[0].node_identifiers)
        self.assertTrue(np.allclose(self.ref_swap,found_unitaries[1].operator))
        self.assertEqual(small_swaps_list[1],
                         found_unitaries[1].node_identifiers)
        ref_op = expm(-1j*delta_t*self.factor1*np.kron(self.operators[0],
                                                      self.operators[1]))
        ref_op = ref_op.reshape(2,2,2,2)
        found_op = found_unitaries[2].operator
        self.assertTrue(np.allclose(ref_op,found_op))
        self.assertEqual(["c1","root"],found_unitaries[2].node_identifiers)
        self.assertTrue(np.allclose(self.ref_swap,found_unitaries[3].operator))
        self.assertEqual(small_swaps_list[0],
                         found_unitaries[3].node_identifiers)
        self.assertTrue(np.allclose(self.ref_swap,found_unitaries[4].operator))
        self.assertEqual(small_swaps_list[1],
                         found_unitaries[4].node_identifiers)
        # Step 2
        ref_op = expm(-1j*delta_t*1*np.kron(self.operators[0],
                                            self.operators[1]))
        ref_op = ref_op.reshape(2,2,2,2)
        found_op = found_unitaries[5].operator
        self.assertTrue(np.allclose(ref_op,found_op))
        self.assertEqual(["c1","root"],found_unitaries[5].node_identifiers)
        # Step 3
        self.assertTrue(np.allclose(self.ref_swap,found_unitaries[6].operator))
        self.assertEqual(small_swaps_list[0],
                         found_unitaries[6].node_identifiers)
        self.assertTrue(np.allclose(self.ref_swap,found_unitaries[7].operator))
        self.assertEqual(small_swaps_list[1],
                         found_unitaries[7].node_identifiers)
        ref_op = expm(-1j*delta_t*self.factor2*np.kron(self.operators[2],
                                                      self.operators[3]))
        ref_op = ref_op.reshape(2,2,2,2)
        found_op = found_unitaries[8].operator
        self.assertTrue(np.allclose(ref_op,found_op))
        self.assertEqual(["root","c2"],found_unitaries[8].node_identifiers)
        self.assertTrue(np.allclose(self.ref_swap,found_unitaries[9].operator))
        self.assertEqual(small_swaps_list[0],
                         found_unitaries[9].node_identifiers)
        self.assertTrue(np.allclose(self.ref_swap,found_unitaries[10].operator))
        self.assertEqual(small_swaps_list[1],
                         found_unitaries[10].node_identifiers)

if __name__ == "__main__":
    unittest.main()
