import sys
sys.path.append('.')

import time
import numpy as np
from pytreenet.util import fast_exp_action


pauli = [np.array([[0, 1], [1, 0]], dtype=complex),
         np.array([[0, -1j], [1j, 0]], dtype=complex),
         np.array([[1, 0], [0, -1]], dtype=complex)]

def test_fast_expm(max_size, iterations):
    sizes = [0]
    times = [0]
    errors = [0]
    
    for i in range(max_size):
        if i>0:
            mat = np.array([1])
            vector = 1 / np.sqrt(2**i) * np.ones((2**i))
            for _ in range(i):
                operator_id = np.random.choice(3, 1)[0]
                mat = np.kron(mat, pauli[operator_id])
            
            _times = []
            _errors = []
            for mode in ["expm", "eigsh", "chebyshev", "none"]:
                start = time.time()
                for _ in range(iterations):
                    res = fast_exp_action(mat, vector, mode)
                _times.append(round(1000*(time.time()-start)/iterations))
                _errors.append(np.round(np.linalg.norm(res - fast_exp_action(mat, vector, mode="expm")), 6))

            times.append(_times)
            errors.append(_errors)
            sizes.append(i)
            print(f"{sizes[i]}\t\t{times[i][0]}\t{errors[i][1]:>8}\t{times[i][1]}\t{errors[i][1]:>8}\t{times[i][2]}\t{errors[i][2]:>8}\t{times[i][3]}\t{errors[i][3]:>8}")

    return sizes, times


if __name__ == "__main__":
    print("Speed testing pytreenet.tdvp.fast_expm():")
    max_size = 10
    iterations = 2
    print("Mean of", iterations, "\n")
    print("log2(size)\ttime expm [ms]\t\ttime eigsh [ms]\t\ttime chebyshev [ms]\ttime err_test [ms]")
    sizes, times = test_fast_expm(max_size, iterations)