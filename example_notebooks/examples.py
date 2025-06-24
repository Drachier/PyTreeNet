"""
This file is used to provide example code for the documentation and
the user guide accompanying the package.

To fit the user guide, the lines should be strictly below 70
characters, compared to the usual 80 characters in the codebase.

However, the file should still be a valid Python file and should not
raise any errors when executed.
"""
import numpy as np

import pytreenet as ptn

# Generating a random tensor
# --------------------------
from pytreenet import random
shape = (2, 3, 4)
tensor = random.crandn(shape)

# QR Decomposition
# ----------------
from pytreenet import util
tensor = random.crandn((2,3,4,5))
qlegs = (0, 1)
rlegs = (2, 3)
Q, R = util.tensor_qr_decomposition(tensor,
                                        qlegs,
                                        rlegs)

Q, R = util.tensor_qr_decomposition(tensor,
                                        qlegs,
                                        rlegs,
                                        mode=ptn.SplitMode.REDUCED)
print(Q.shape) # -> (2,3,6)
print(R.shape) # -> (6,4,5)

Q, R = util.tensor_qr_decomposition(tensor,
                                        qlegs,
                                        rlegs,
                                        mode=ptn.SplitMode.FULL)
print(Q.shape) # -> (2,3,6)
print(R.shape) # -> (6,4,5)

Q, R = util.tensor_qr_decomposition(tensor,
                                        qlegs,
                                        rlegs,
                                        mode=ptn.SplitMode.KEEP)
print(Q.shape) # -> (2,3,20)
print(R.shape) # -> (20,4,5)

# SVD Decomposition
# -----------------
svector = [1,0.5,0.05,0]
tensor = np.diag(svector).reshape(2,2,2,2)
u_legs = (0,1)
v_legs = (2,3)

U, S, Vh = util.tensor_svd(tensor, u_legs, v_legs,
                               mode=ptn.SplitMode.REDUCED)
print(S.shape) # -> (3, )

U, S, Vh = util.tensor_svd(tensor, u_legs, v_legs,
                               mode=ptn.SplitMode.FULL)
print(S.shape) # -> (4, )

# Truncated SVD Decomposition
# ---------------------------
svd_params = util.SVDParameters()
svd_params.total_tol = 1e-2
U, S, Vh = util.truncated_tensor_svd(tensor, u_legs, v_legs,
                                         svd_params=svd_params)
print(S.shape) # -> (3, )

svd_params.rel_tol = 1e-1
U, S, Vh = util.truncated_tensor_svd(tensor, u_legs, v_legs,
                                         svd_params=svd_params)
print(S.shape) # -> (2, )

svd_params.max_bond_dim = 1
U, S, Vh = util.truncated_tensor_svd(tensor, u_legs, v_legs,
                                         svd_params=svd_params)
print(S.shape) # -> (1, )

# Building a TTN
# --------------
ttn = ptn.TreeTensorNetwork()

root_node, root_tensor = random.random_tensor_node((2,4,5,3),
                                                       "root")
ttn.add_root(root_node, root_tensor)

node1, tensor1 = random.random_tensor_node((4,2,2),"node1")
ttn.add_child_to_parent(node1, tensor1, 0, "root", 1)

print(node1.neighbour_index("root")) # -> 0
print(root_node.neighbour_index("node1")) # -> 0

node4, tensor4 = random.random_tensor_node((5,2,3),"node4")
node5, tensor5 = random.random_tensor_node((3,2),"node5")
ttn.add_child_to_parent(node4, tensor4, 0, "root", 2)
ttn.add_child_to_parent(node5, tensor5, 0, "root", 3)

node2, tensor2 = random.random_tensor_node((2,2),"node2")
node3, tensor3 = random.random_tensor_node((2,2),"node3")
node6, tensor6 = random.random_tensor_node((2,2),"node6")
ttn.add_child_to_parent(node2, tensor2, 0, "node1", 1)
ttn.add_child_to_parent(node3, tensor3, 0, "node1", 2)
ttn.add_child_to_parent(node6, tensor6, 0, "node5", 1)

# Accesing a Node
# --------------
node = ttn.nodes["node1"]
tensor = ttn.tensors["node1"]
node_s, tensor_s = ttn["node1"]

# Accessomg the Root
# ------------------
root_id = ttn.root_id
root_node, root_tensor = ttn[root_id]
root_node, root_tensor = ttn.root

# Contracting two Nodes
# ---------------------
ttn.contract_nodes("node5", "node6", "new")
print(len(ttn.nodes)) # -> 6
print(ttn.nodes["new"].shape) # -> (3,2)


# Contracting the whole TTN
# -------------------------
result, order = ttn.completely_contract_tree(to_copy=True)
print(result.shape) # -> (2,2,2,2,3,2)
print(order)
# -> ["root", "node1", "node2", "node3", "node4", "new"]

# Creating the smallest non-trivial Contraction
ttn_dash = ptn.TreeTensorNetwork()
p_node, p_tensor = random.random_tensor_node((3,),"p")
ttn_dash.add_root(p_node, p_tensor)
a_node, a_tensor = random.random_tensor_node((3,4,5,2),"a")
ttn_dash.add_child_to_parent(a_node, a_tensor, 0,
                             "p", 0)
c1_node, c1_tensor = random.random_tensor_node((4,),"c1")
ttn_dash.add_child_to_parent(c1_node, c1_tensor, 0,
                             "a", 1)
b_node, b_tensor = random.random_tensor_node((5,4,3),"b")
ttn_dash.add_child_to_parent(b_node, b_tensor, 0,
                             "a", 2)
c2_node, c2_tensor = random.random_tensor_node((4,),"c2")
ttn_dash.add_child_to_parent(c2_node, c2_tensor, 0,
                             "b", 1)

ttn_dash.contract_nodes("a", "b", "ab")

# Splitting
#----------
ab_node = ttn_dash.nodes["ab"]
a_legs = ptn.LegSpecification("p",["c1"],[3],node=ab_node)
b_legs = ptn.LegSpecification(None,["c2"],[4],node=ab_node)

ttn_dash.split_node_qr("ab",a_legs,b_legs,
                        q_identifier="a",
                        r_identifier="b")

# Splitting a Root
#-----------------
root_id = ttn.root_id
node0 = ttn.nodes[root_id]
legs_01 = ptn.LegSpecification(None,["node1","node4"],[],
                               node=node0,is_root=True)
legs_02 = ptn.LegSpecification(None,["new"],[node0.nvirt_legs()],
                               node=node0)
ttn.split_node_qr(root_id,legs_01,legs_02,
                   q_identifier="node01",
                   r_identifier="node02")

# The Canonical Form
# ------------------
## Rebuilding the example TTN
ttn = ptn.TreeTensorNetwork()
root_node, root_tensor = random.random_tensor_node((2,4,5,3),
                                                       "root")
ttn.add_root(root_node, root_tensor)
node1, tensor1 = random.random_tensor_node((4,2,2),"node1")
ttn.add_child_to_parent(node1, tensor1, 0, "root", 1)
node4, tensor4 = random.random_tensor_node((5,2,3),"node4")
node5, tensor5 = random.random_tensor_node((3,2),"node5")
ttn.add_child_to_parent(node4, tensor4, 0, "root", 2)
ttn.add_child_to_parent(node5, tensor5, 0, "root", 3)
node2, tensor2 = random.random_tensor_node((2,2),"node2")
node3, tensor3 = random.random_tensor_node((2,2),"node3")
node6, tensor6 = random.random_tensor_node((2,2),"node6")
ttn.add_child_to_parent(node2, tensor2, 0, "node1", 1)
ttn.add_child_to_parent(node3, tensor3, 0, "node1", 2)
ttn.add_child_to_parent(node6, tensor6, 0, "node5", 1)

## Canonicalizing the TTN
ttn.canonical_form("root")

# Move the Orthogonality Centre
# -----------------------------
ttn.move_orthogonalization_center("node6")

# Example TTNS
# ------------
ttns = ptn.TreeTensorNetworkState()
center_node = ptn.Node(identifier="0")
center_tensor = random.crandn((4,4,4,2))
ttns.add_root(center_node, center_tensor)
for i in range(3):
    chain_node = ptn.Node(identifier=f"{i}0")
    chain_tensor = random.crandn((4,3,2))
    ttns.add_child_to_parent(chain_node, chain_tensor,
                                0,"0",i)
    end_node = ptn.Node(identifier=f"{i}1")
    end_tensor = random.crandn((3,2))
    ttns.add_child_to_parent(end_node, end_tensor,
                                0,f"{i}0",1)

# Tensor Product
# --------------
operators = {f"{i}0": random.random_hermitian_matrix(2)}
omega = ptn.TensorProduct(operators)

# Expectation Values
# ------------------
sc_prod = ttns.operator_expectation_value(ptn.TensorProduct())
z20 = ptn.TensorProduct({"20": operators["20"]})
single_site_exp = ttns.operator_expectation_value(z20)
operator_exp = ttns.operator_expectation_value(z20)

# Example TTNO
# ------------
ttno = ptn.TTNO()
center_node = ptn.Node(identifier="0")
center_tensor = random.crandn((4,4,4,2,2))
ttno.add_root(center_node, center_tensor)
for i in range(3):
    chain_node = ptn.Node(identifier=f"{i}0")
    chain_tensor = random.crandn((4,3,2,2))
    ttno.add_child_to_parent(chain_node, chain_tensor,
                                0,"0",i)
    end_node = ptn.Node(identifier=f"{i}1")
    end_tensor = random.crandn((3,2,2))
    ttno.add_child_to_parent(end_node, end_tensor,
                                0,f"{i}0",1)

# TTNO TTNS Exp Values
# --------------------
exp_val = ttns.operator_expectation_value(ttno)

# Hamiltonian
# -----------
term1 = ptn.TensorProduct({"00":"P1", "10": "P0", "20": "P1"})
term2 = ptn.TensorProduct({"00":"P0", "10": "P1", "20": "P0"})
term3 = ptn.TensorProduct({"00":"P0", "10": "P0", "20": "P1"})
p0 = np.asarray([[1,0],[0,0]], dtype=complex)
p1 = np.asarray([[0,0],[0,1]], dtype=complex)
conversion_dict = {"P0": p0, "P1": p1,
                   "I1": np.eye(1), "I2": np.eye(2)}
ham = ptn.Hamiltonian([term1, term2, term3], conversion_dict)

# Auto TTNO Generation
# --------------------
ttno = ptn.TreeTensorNetworkOperator.from_hamiltonian(ham,ttns)

# First Order Suzuki-Trotter splitting
# ------------------------------------
qubits = ["q1", "q2"]
dim = 2
A  = ptn.TensorProduct({idt: random.random_hermitian_matrix(dim)
                        for idt in qubits})
B  = ptn.TensorProduct({idt: random.random_hermitian_matrix(dim)
                        for idt in qubits})
stepA = ptn.TrotterStep(A,1)
stepB = ptn.TrotterStep(B,1)
fstorder_ST = ptn.TrotterSplitting([stepA,stepB])
delta_time = 0.01
unitaries_1oST = fstorder_ST.exponentiate_splitting(delta_time,
                                                    dim=dim)

# Strang Splitting
# ----------------
stepA = ptn.TrotterStep(A,0.5)
stepB = ptn.TrotterStep(B,1)
strang = ptn.TrotterSplitting([stepA,stepB,stepA])
unitaries_Strang = strang.exponentiate_splitting(delta_time,
                                                 dim=dim)

# Complicated Splitting
# ---------------------
ttns = ptn.TreeTensorNetworkState()
center_node = ptn.Node(identifier="0")
center_tensor = random.crandn((4,4,4,2))
ttns.add_root(center_node, center_tensor)
for i in range(3):
    chain_node = ptn.Node(identifier=f"{i}0")
    chain_tensor = random.crandn((4,3,2))
    ttns.add_child_to_parent(chain_node, chain_tensor,
                                0,"0",i)
    end_node = ptn.Node(identifier=f"{i}1")
    end_tensor = random.crandn((3,2))
    ttns.add_child_to_parent(end_node, end_tensor,
                                0,f"{i}0",1)

steps = []
X, _, _ = ptn.operators.pauli_matrices()
for ident, node in ttns.nodes.items():
    if not node.is_root():
        op = ptn.TensorProduct({ident: X,
                                node.parent: X})
        step = ptn.TrotterStep(op, 1)
        steps.append(step)

nnn_op = ptn.TensorProduct({"00": X, "20": X})
swaps = ptn.SWAPlist([("00","0")])
nnn_step = ptn.TrotterStep(nnn_op, 1,
                           swaps_before=swaps,
                           swaps_after=swaps)
steps.append(nnn_step)

# TEBD
from copy import deepcopy
# ----
state = np.zeros((2, ))
zero_state = deepcopy(state)
zero_state[0] = 1
one_state = deepcopy(state)
one_state[1] = 1
ttns = ptn.TreeTensorNetworkState()
center_node = ptn.Node(identifier="0")
center_tensor = deepcopy(zero_state.reshape(1,1,1,2))
ttns.add_root(center_node, center_tensor)
for i in range(3):
    chain_node = ptn.Node(identifier=f"{i}0")
    chain_tensor = deepcopy(one_state.reshape(1,1,2))
    ttns.add_child_to_parent(chain_node, chain_tensor,
                                0,"0",i)
    end_node = ptn.Node(identifier=f"{i}1")
    end_tensor = deepcopy(zero_state.reshape(1,2))
    ttns.add_child_to_parent(end_node, end_tensor,
                                0,f"{i}0",1)

## Trotterisation
X, _, Z = ptn.operators.pauli_matrices()
g = -0.1
gX = g*X
steps = []
for ident, node in ttns.nodes.items():
    tp1 = ptn.TensorProduct({ident: gX})
    tp1 = ptn.TrotterStep(tp1, 1)
    steps.append(tp1)
    if not node.is_root():
        op = ptn.TensorProduct({ident: -1*Z,
                                node.parent: Z})
        tp = ptn.TrotterStep(op, 1)
        steps.append(tp)
trotter = ptn.TrotterSplitting(steps)
delta_t = 0.01
magn = ptn.TensorProduct({ide: Z for ide in ttns.nodes})
final_time = 1
max_bd = 3
svd_params = util.SVDParameters(max_bond_dim=max_bd,
                                    rel_tol=1e-5,
                                    total_tol=1e-6)
config = ptn.TTNTimeEvolutionConfig(record_bond_dim=True)
tebd = ptn.TEBD(ttns, trotter,
                delta_t, final_time,
                {"magn": magn}, svd_params,
                config=config)
tebd.run()

## Sanity Checks
times = tebd.times()
print(times.shape) # -> (101, )
bond_dims = tebd.operator_result("bond_dim")
print(len(bond_dims)) # -> 6
print(len(bond_dims[("0","00")])) # -> 101
print(tebd.results_real()) # -> True
magnetization = tebd.operator_result("magn", realise=True)
print(magnetization.shape) # -> (101, )


# 1TDVP
# ------

state = np.zeros((2, ))
zero_state = deepcopy(state)
zero_state[0] = 1
one_state = deepcopy(state)
one_state[1] = 1
ttns = ptn.TreeTensorNetworkState()
center_node = ptn.Node(identifier="0")
bond_dim = 3
center_tensor = np.asarray([1]).reshape(1,1,1,2)
center_tensor = np.pad(center_tensor,
                       ((0, bond_dim-1),(0, bond_dim-1),
                        (0, bond_dim-1),(0, 0)))
ttns.add_root(center_node, center_tensor)
for i in range(3):
    chain_node = ptn.Node(identifier=f"{i}0")
    chain_tensor = deepcopy(one_state.reshape(1,1,2))
    chain_tensor = np.pad(chain_tensor, 
                          ((0, bond_dim-1),(0, bond_dim-1),(0, 0)))
    ttns.add_child_to_parent(chain_node, chain_tensor,
                                0,"0",i)
    end_node = ptn.Node(identifier=f"{i}1")
    end_tensor = deepcopy(zero_state.reshape(1,2))
    end_tensor = np.pad(end_tensor, ((0, bond_dim-1), (0, 0)))
    ttns.add_child_to_parent(end_node, end_tensor,
                                0,f"{i}0",1)

X, _, Z = ptn.operators.pauli_matrices()
g = -0.1
gX = g*X
con_dict = {"gX": gX, "Z": Z, "-Z": -1*Z,
                    "I1": np.eye(1), "I2": np.eye(2)}
ham = ptn.operators.Hamiltonian(conversion_dictionary=con_dict)
for ident, node in ttns.nodes.items():
    if not node.is_root():
        op = ptn.TensorProduct({ident: "gX"})
        ham.add_term(op)
        if node.parent != ttns.root_id:
            op = ptn.TensorProduct({ident: "-Z",
                                    node.parent: "Z"})
            ham.add_term(op)
triple_term = ptn.TensorProduct({"00": "-Z", "10": "Z", "20": "Z"})
ham.add_term(triple_term)
ham_pad = ham.pad_with_identities(ttns)
ttno = ptn.TreeTensorNetworkOperator.from_hamiltonian(ham_pad,
                                                      ttns)
magn = ptn.TensorProduct({ide: Z for ide in ttns.nodes
                          if ide != ttns.root_id})
ops = {"magn": magn}

delta_t = 0.01
final_time = 1
config = ptn.TTNTimeEvolutionConfig(record_bond_dim=True)
tdvp_onesite = ptn.FirstOrderOneSiteTDVP(ttns, ttno,
                                         delta_t, final_time,
                                         ops,
                                         config=config)
tdvp_onesite.run()
