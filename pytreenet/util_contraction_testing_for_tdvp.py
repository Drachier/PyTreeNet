import numpy as np


class TensorNetwork:
    def __init__(self):
        self._nodes = dict()
        self._edges = None

    @property
    def nodes(self):
        return self._nodes
    
    def add_node(self, new_entry):
        self._nodes.update(new_entry)

    @property
    def num_contractions(self):
        return self.num_connected_legs/2

    @property
    def num_connected_legs(self):
        num_connected_legs = 0
        for node_id in self.nodes.keys():
            num_connected_legs += len(self.nodes[node_id].connected_legs.keys())
        return num_connected_legs
    
    @property
    def num_open_legs(self):
        num_open_legs = 0
        for node_id in self.nodes.keys():
            num_open_legs += len(self.nodes[node_id].open_legs)
        return num_open_legs
    
    @property
    def edges(self):
        if self._edges is None:
            node_ids_completed = []
            edges = []
            for node1_id in self.nodes.keys():
                for node2_id in self.nodes[node1_id].connected_legs.keys():
                    if node2_id not in node_ids_completed:
                        edges.append(Edge(node1_id, node2_id, self.nodes[node1_id].connected_legs[node2_id], self.nodes[node2_id].connected_legs[node1_id]))
                node_ids_completed.append(node1_id)
            self._edges = edges
            return edges
        else:
            return self._edges
    
    def contract_except(self, exceptions: list):
        edges = []
        for edge in self.edges:
            if not (edge.name1 in exceptions or edge.name2 in exceptions):
                edges.append(edge)
        self.contract_edges(edges)
        pass
    
    def contract_edges(self, edges: list):
        for edge in edges:
            self.contract_edge(edge)
    
    def contract_edge(self, edge):
        pass

    def contract_any_two_nodes(self):
        node1 = None
        while node1 == None:
            for key in self.nodes.keys():
                if len(self.nodes[key].connected_legs.keys()) > 0:
                    node1 = key
                    node2 = list(self.nodes[key].connected_legs.keys())[0]
        
        self.contract_two_nodes(node1, node2)

    def contract_two_nodes(self, node1_id, node2_id):
        tensor1 = self.nodes[node1_id].tensor
        tensor2 = self.nodes[node2_id].tensor

        legs_node1_to_node2 = self.nodes[node1_id].connected_legs[node2_id]
        if type(legs_node1_to_node2) is int:
            num_uncontracted_legs_node1 = tensor1.ndim - 1
        else:
            num_uncontracted_legs_node1 = tensor1.ndim - len(legs_node1_to_node2)

        legs_node2_to_node1 = self.nodes[node2_id].connected_legs[node1_id]

        new_tensor = np.tensordot(tensor1, tensor2, axes=(legs_node1_to_node2, legs_node2_to_node1))

        new_node_id = node1_id + "_" + node2_id
        new_node = Node(new_tensor)
        
        # create new node connected legs
        for node_id in [node1_id, node2_id]:
            for key in self.nodes[node_id].connected_legs.keys():
                connected_with_both = False
                if key != new_node_id:
                    new_leg = self.nodes[node_id].connected_legs[key] + (node_id == node2_id) * (num_uncontracted_legs_node1 - 1)
                    if key not in new_node.connected_legs.keys():
                        new_node.connected_legs.update({key: new_leg})
                    else:
                        # second connection incoming
                        connected_with_both = True
                        new_node.connected_legs.update({key: [new_node.connected_legs[key], new_leg]})
    
                    # replace in connected_legs of neighbours
                    if node_id in self.nodes[key].connected_legs.keys():
                        if not connected_with_both:
                            self.nodes[key].connected_legs[new_node_id] = self.nodes[key].connected_legs[node_id]
                        else:
                            if type(self.nodes[key].connected_legs[node1_id]) is int:
                                legs1 = [self.nodes[key].connected_legs[node1_id]]
                            else:
                                legs1 = self.nodes[key].connected_legs[node1_id]
                            if type(self.nodes[key].connected_legs[node2_id]) is int:
                                legs2 = [self.nodes[key].connected_legs[node1_id]]
                            else:
                                legs2 = self.nodes[key].connected_legs[node1_id]
                            self.nodes[key].connected_legs[new_node_id] = legs1 + legs2
                        self.nodes[key].connected_legs.pop(node_id)

        self.nodes.pop(node1_id)
        self.nodes.pop(node2_id)
        self.add_node({new_node_id: new_node})
        
    
class Edge(object):
    def __init__(self, name1, name2, leg1, leg2):
        self.name1 = name1
        self.name2 = name2
        self.leg1 = leg1
        self.leg2 = leg2


class Node(object):
    def __init__(self, tensor):

        self._tensor = np.asarray(tensor)

        #At the beginning all legs are open
        self._open_legs = list(np.arange(tensor.ndim))

        self.connected_legs = dict()

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, new_tensor):
        self._tensor = new_tensor
    
    @property
    def open_legs(self):
        self._open_legs = []
        for i in list(np.arange(self._tensor.ndim)):
            if i not in self.connected_legs.values():
                self._open_legs.append(i)
        return self._open_legs