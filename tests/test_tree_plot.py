import pytreenet as ptn

class TestTreePlot:
    def __init__(self):
        self.initialize_ttn()
        ptn.ttn_plot.plot(self.ttn)
    
    def initialize_ttn(self):
        self.ttn = ptn.TreeTensorNetwork()

        node1 = ptn.random_tensor_node((2,2), identifier="node1")
        node2 = ptn.random_tensor_node((2,3,3,3), identifier="node2")
        node3 = ptn.random_tensor_node((3,5), identifier="node3")
        node4 = ptn.random_tensor_node((3,4), identifier="node4")
        node5 = ptn.random_tensor_node((2,3,3), identifier="node5")
        node6 = ptn.random_tensor_node((3,4,4), identifier="node6")
        node7 = ptn.random_tensor_node((4,2,2), identifier="node7")
        node8 = ptn.random_tensor_node((3,2,2), identifier="node8")
        node9 = ptn.random_tensor_node((5,2,2), identifier="node9")

        self.ttn.add_root(node1)
        self.ttn.add_child_to_parent(node2, 0, "node1", 0)
        self.ttn.add_child_to_parent(node3, 0, "node2", 1)
        self.ttn.add_child_to_parent(node4, 0, "node2", 2)
        self.ttn.add_child_to_parent(node5, 0, "node1", 1)
        self.ttn.add_child_to_parent(node6, 0, "node5", 2)
        self.ttn.add_child_to_parent(node7, 0, "node6", 1)
        self.ttn.add_child_to_parent(node8, 0, "node2", 3)
        self.ttn.add_child_to_parent(node9, 0, "node3", 1)

