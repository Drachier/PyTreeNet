import unittest
import numpy as np
from copy import deepcopy
from fractions import Fraction

from pytreenet.ttno.symbolic_gaussian_elimination_fraction import gaussian_elimination

class TestGaussianElimination(unittest.TestCase):
    def test_simple_all_int_matrix(self):
        matrix = [
            [2, 1, -1],
            [-3, -1, 2],
            [-2, 1, 2]
        ]
        matrix_f = [[Fraction(x) for x in row] for row in matrix]

        Op_l,Gamma_m, Op_r = gaussian_elimination(deepcopy(matrix_f))
        #self.assertEqual(result, matrix)

        Op_l_array = np.array(Op_l)
        Gamma_m_array = np.array(Gamma_m)
        Op_r_array = np.array(Op_r)
        result = Op_l_array @ Gamma_m_array @ Op_r_array
        result_i = [[int(x) for x in row] for row in result]
        self.assertTrue(np.allclose(np.array(matrix), result_i))

    
    def test_rectangular_all_int_matrix(self):
        matrix = [
            [2, 1, -1, 5, 7],
            [-3, -1, 2, 1, 0],
            [-2, 1, 2, -2, 6]
        ]
        matrix_f = [[Fraction(x) for x in row] for row in matrix]

        Op_l,Gamma_m, Op_r = gaussian_elimination(deepcopy(matrix_f))

        #self.assertEqual(result, matrix)

        Op_l_array = np.array(Op_l)
        Gamma_m_array = np.array(Gamma_m)
        Op_r_array = np.array(Op_r)
        result = Op_l_array @ Gamma_m_array @ Op_r_array
        result_i = [[int(x) for x in row] for row in result]

        self.assertTrue(np.allclose(np.array(matrix), result_i))
    
    def test_simple_gamma_matrix(self):
        matrix = [[(1,"a"), (1, "b"), (1,"c"), 0, (1,"b")], [0, (1,"d"), 0, 0, (1,"d")], [0,(1,"e"), 0, 0, (1,"e")], [0, 0, (1,"f"), (1,"g"),0], [0, 0, (1,"f"), (1,"g"),0]]
        matrix_f = [[ Fraction(item) if isinstance(item, int) else (Fraction(item[0]), item[1]) if isinstance(item, tuple) else item for item in row]for row in matrix]

        true_Op_l = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]]
        true_Op_l_f = [[Fraction(x) for x in row] for row in true_Op_l]
        true_gamma = [[(1, 'a'), (1, 'b'), (1, 'c'), 0], [0, (1, 'd'), 0, 0], [0, 0, (1, 'f'), (1, 'g')], [0, (1, 'e'), 0, 0]]
        true_gamma_f = [[ Fraction(item) if isinstance(item, int) else (Fraction(item[0]), item[1]) if isinstance(item, tuple) else item for item in row]for row in true_gamma]
        true_Op_r = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]
        true_Op_r_f = [[Fraction(x) for x in row] for row in true_Op_r]
        Op_l,Gamma_m, Op_r = gaussian_elimination(matrix_f)


        self.assertEqual(Op_l, true_Op_l_f)
        self.assertEqual(Gamma_m, true_gamma_f)
        self.assertEqual(Op_r, true_Op_r_f)

    def test_row_eliminated_gamma_matrix(self):
        matrix = [[(1,"a"), (1, "b"), 0, 0], [0, (1,"b"), (1,"c"), 0], [(1,"a"), 0, 0, (1,"d")], [0, 0, (1,"c"), (1,"d")]]
        matrix_f = [[ Fraction(item) if isinstance(item, int) else (Fraction(item[0]), item[1]) if isinstance(item, tuple) else item for item in row]for row in matrix]

        true_Op_l = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        true_Op_l_f = [[Fraction(x) for x in row] for row in true_Op_l]
        
        true_gamma = [[(1, 'a'), 0, 0, (1.0, 'd')], [0, (1, 'b'), 0, (-1.0, 'd')], [0, 0, (1.0, 'c'), (1, 'd')]]
        true_gamma_f = [[ Fraction(item) if isinstance(item, int) else (Fraction(item[0]), item[1]) if isinstance(item, tuple) else item for item in row]for row in true_gamma]
        
        true_Op_r = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        true_Op_r_f = [[Fraction(x) for x in row] for row in true_Op_r]

        Op_l,Gamma_m, Op_r = gaussian_elimination(matrix_f)

        self.assertEqual(Op_l, true_Op_l_f)
        self.assertEqual(Gamma_m, true_gamma_f)
        self.assertEqual(Op_r, true_Op_r_f)

if __name__ == "__main__":
    unittest.main()