import numpy as np
import sys
sys.path.append('.')
from pytreenet.tensor_util import tensor_qr_decomposition, tensor_qr_original_shape

"""
Two-node part of a contraction network, e.g. first/last two nodes of an mps.
Goal: QR-decomp on 'a', e.g. for orthogonalization of TTN.
"""
#
#   2 |      2 |
#     |        |
#  ---b---  ---a
#   3   3    3
#

# This is the ordering of legs that is effectivels used when the 3-dim leg of 'a' is used 
# as 'r-legs' in pytreenet.tensor_util.tensor_qr_decomposition as demonstrated below.
a = np.zeros((2, 3))
a[0, 0] = 1
b = np.zeros((2, 3, 3))
b[0, 0, 0] = 1

q_numpy, r_numpy = np.linalg.qr(a, mode='complete')  
# Note: mode='full' is deprecated as of Numpy 1.8.0, use mode='complete'.
# Using 'full' might now lead to incorrect results.
# I have already changed this in pytreenet.tensor_util.tensor_qr_decomposition,
# but this is only a local change until I make a pull request for it.
try:
    q_ptn, r_ptn = tensor_qr_decomposition(a, [0], [1], mode='full')
except:
    q_ptn, r_ptn = tensor_qr_decomposition(a, [0], [1], mode='complete')

assert np.allclose(q_numpy, q_ptn)
assert np.allclose(r_numpy, r_ptn)

assert q_ptn.shape == (2, 2)
assert r_ptn.shape == (2, 3)

"""
Insert 'q' and 'r' into the network. Indicate absorption of 'r' into 'b' as new tensor 'c'.
"""

#
#   2 |             2 |           2|      2|
#     |               |            |       |
#  ---b--- ---r--- ---q     =   ---c--- ---q
#   3   3   3   2   2            3   2   2
#

"""
The virtual bond dimension has changed using this approach.
Introduce zero-padding to the QR decomposition.
"""

q, r = tensor_qr_original_shape(a, [0], [1])

print(q.shape, r.shape)

#
#   2 |             2 |     
#     |               |        
#  ---b--- ---r--- ---q     
#   3   3   3   3   3          
#

