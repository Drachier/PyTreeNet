import unittest
import numpy as np
from copy import deepcopy

import pytreenet as ptn


class TestGaussianElimination(unittest.TestCase):
    def test_simple_all_int_matrix(self):
        matrix = [
            [2, 1, -1],
            [-3, -1, 2],
            [-2, 1, 2]
        ]
        Op_l,Gamma_m, Op_r = ptn.gaussian_elimination(deepcopy(matrix))
        #self.assertEqual(result, matrix)

        Op_l_array = np.array(Op_l)
        Gamma_m_array = np.array(Gamma_m)
        Op_r_array = np.array(Op_r)
        result = Op_l_array @ Gamma_m_array @ Op_r_array
        self.assertTrue(np.allclose(np.array(matrix), result))

    
    def test_rectangular_all_int_matrix(self):
        matrix = [
            [2, 1, -1, 5, 7],
            [-3, -1, 2, 1, 0],
            [-2, 1, 2, -2, 6]
        ]
        Op_l,Gamma_m, Op_r = ptn.gaussian_elimination(deepcopy(matrix))
        #self.assertEqual(result, matrix)

        Op_l_array = np.array(Op_l)
        Gamma_m_array = np.array(Gamma_m)
        Op_r_array = np.array(Op_r)
        result = Op_l_array @ Gamma_m_array @ Op_r_array
        self.assertTrue(np.allclose(np.array(matrix), result))
    
    def test_simple_gamma_matrix(self):
        matrix = [[(1,"a"), (1, "b"), (1,"c"), 0, (1,"b")], [0, (1,"d"), 0, 0, (1,"d")], [0,(1,"e"), 0, 0, (1,"e")], [0, 0, (1,"f"), (1,"g"),0], [0, 0, (1,"f"), (1,"g"),0]]
        
        true_Op_l = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]]
        true_gamma = [[(1, 'a'), (1, 'b'), (1, 'c'), 0], [0, (1, 'd'), 0, 0], [0, 0, (1, 'f'), (1, 'g')], [0, (1, 'e'), 0, 0]]
        true_Op_r = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]

        Op_l,Gamma_m, Op_r = ptn.gaussian_elimination(matrix)

        self.assertEqual(Op_l, true_Op_l)
        self.assertEqual(Gamma_m, true_gamma)
        self.assertEqual(Op_r, true_Op_r)

    def test_row_eliminated_gamma_matrix(self):
        matrix = [[(1,"a"), (1, "b"), 0, 0], [0, (1,"b"), (1,"c"), 0], [(1,"a"), 0, 0, (1,"d")], [0, 0, (1,"c"), (1,"d")]]
        
        true_Op_l = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        true_gamma = [[(1, 'a'), 0, 0, (1.0, 'd')], [0, (1, 'b'), 0, (-1.0, 'd')], [0, 0, (1.0, 'c'), (1, 'd')]]
        true_Op_r = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        Op_l,Gamma_m, Op_r = ptn.gaussian_elimination(matrix)

        self.assertEqual(Op_l, true_Op_l)
        self.assertEqual(Gamma_m, true_gamma)
        self.assertEqual(Op_r, true_Op_r)

if __name__ == "__main__":
    unittest.main()