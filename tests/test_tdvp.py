import pytreenet as ptn
import numpy as np

class TestTDVP:
    def __init__(self):
        self.initialize_ttn()
        self.initialize_hamiltonian()

        self.ttn.plot("State > TreeTensorState")
        self.hamiltonian.plot("Hamiltonian > TreeTensorOperator")

        self.tts = ptn.TreeTensorState(self.ttn)
        self.tts.adjoint()

        self.ttn = ptn.tdvp(self.ttn, self.hamiltonian, .5, 1, "1site")
    
    def initialize_ttn(self):
        self.ttn = ptn.TreeTensorNetwork()

        node0 = ptn.random_tensor_node((2,2,2), identifier="node0")
        node1 = ptn.random_tensor_node((2,2,2,2), identifier="node1")
        node2 = ptn.random_tensor_node((2,2,2), identifier="node2")
        node3 = ptn.random_tensor_node((2,2), identifier="node3")
        node4 = ptn.random_tensor_node((2,2), identifier="node4")
        node5 = ptn.random_tensor_node((2,2,2), identifier="node5")
        node6 = ptn.random_tensor_node((2,2,2,2,2), identifier="node6")
        node7 = ptn.random_tensor_node((2,2), identifier="node7")
        node8 = ptn.random_tensor_node((2,2), identifier="node8")
        node9 = ptn.random_tensor_node((2,2), identifier="node9")

        self.ttn.add_root(node0)

        self.ttn.add_child_to_parent(node1, 0, "node0", 0)

        self.ttn.add_child_to_parent(node3, 0, "node1", 1)
        self.ttn.add_child_to_parent(node4, 0, "node1", 2)

        self.ttn.add_child_to_parent(node2, 0, "node0", 1)

        self.ttn.add_child_to_parent(node5, 0, "node2", 1)

        self.ttn.add_child_to_parent(node6, 0, "node5", 1)

        self.ttn.add_child_to_parent(node7, 0, "node6", 1)
        self.ttn.add_child_to_parent(node8, 0, "node6", 2)
        self.ttn.add_child_to_parent(node9, 0, "node6", 3)
    
    def initialize_hamiltonian(self):
        self.hamiltonian = ptn.TreeTensorOperator()

        node0 = ptn.random_tensor_node((2,2,2,2), identifier="node0")
        node1 = ptn.random_tensor_node((2,2,2,2,2), identifier="node1")
        node2 = ptn.random_tensor_node((2,2,2,2), identifier="node2")
        node3 = ptn.random_tensor_node((2,2,2), identifier="node3")
        node4 = ptn.random_tensor_node((2,2,2), identifier="node4")
        node5 = ptn.random_tensor_node((2,2,2,2), identifier="node5")
        node6 = ptn.random_tensor_node((2,2,2,2,2,2), identifier="node6")
        node7 = ptn.random_tensor_node((2,2,2), identifier="node7")
        node8 = ptn.random_tensor_node((2,2,2), identifier="node8")
        node9 = ptn.random_tensor_node((2,2,2), identifier="node9")

        self.hamiltonian.add_root(node0)

        self.hamiltonian.add_child_to_parent(node1, 1, "node0", 1)

        self.hamiltonian.add_child_to_parent(node3, 1, "node1", 2)
        self.hamiltonian.add_child_to_parent(node4, 1, "node1", 3)

        self.hamiltonian.add_child_to_parent(node2, 1, "node0", 2)

        self.hamiltonian.add_child_to_parent(node5, 1, "node2", 2)

        self.hamiltonian.add_child_to_parent(node6, 1, "node5", 2)

        self.hamiltonian.add_child_to_parent(node7, 1, "node6", 2)
        self.hamiltonian.add_child_to_parent(node8, 1, "node6", 3)
        self.hamiltonian.add_child_to_parent(node9, 1, "node6", 4)

        self.hamiltonian.add_link("node0", "node0")
        self.hamiltonian.add_link("node1", "node1")
        self.hamiltonian.add_link("node2", "node2")
        self.hamiltonian.add_link("node3", "node3")
        self.hamiltonian.add_link("node4", "node4")
        self.hamiltonian.add_link("node5", "node5")
        self.hamiltonian.add_link("node6", "node6")
        self.hamiltonian.add_link("node7", "node7")
        self.hamiltonian.add_link("node8", "node8")
        self.hamiltonian.add_link("node9", "node9")

