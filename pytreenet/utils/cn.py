import numpy as np


class Node:
    def __init__(self, id, tensor):
        self.id = id
        self.tensor = tensor
        self.connected_legs = dict()
        self.contraction_history = []
    
    def group_parallel_legs(self):
        legs = self.connected_legs.values()
        leg_order = [i for sublist in legs for i in (sublist if type(sublist)==tuple else (sublist,))]
        self.tensor = self.tensor.transpose(leg_order)
        new_shape = self.tensor.shape
        new_shape_should_be = get_new_shape(new_shape, legs)
        self.tensor = self.tensor.reshape(new_shape_should_be)
        for i, node_id in enumerate(self.connected_legs.keys()):
            self.connected_legs[node_id] = i
    

def get_new_shape(shape, legs):
    new_shape = [None] * len(legs)
    current_index = 0
    for i, l in enumerate(legs):
        if type(l)==tuple:
            new_shape[i] = shape[current_index]*shape[current_index+1]
            current_index += 2
        else:
            new_shape[i] = shape[current_index]
            current_index += 1
    return new_shape


class QNode(Node):
    def __init__(self, tt_node, type, connection=None) -> None:
        self.contraction_history = []
        
        self.id = type + "_" + tt_node.identifier
        self.tensor = tt_node.tensor if type!="bra" else tt_node.tensor.conj()

        connected_legs = tt_node.neighbouring_nodes()
        self.connected_legs = dict()
        for node_id in connected_legs.keys():
            self.connected_legs[type + "_" + node_id] = connected_legs[node_id]
        if type in ["bra", "ket"] and connection is not None:
            self.connected_legs[connection + "_" + tt_node.identifier] = tt_node.physical_leg
        elif type=="ham":
            self.connected_legs["bra_" + tt_node.identifier] = tt_node.physical_leg_bra
            self.connected_legs["ket_" + tt_node.identifier] = tt_node.physical_leg_ket
        else:
            self.connected_legs["open_" + tt_node.identifier] = tt_node.physical_leg


class ContractionNetwork:
    def __init__(self):
        self.nodes = dict()
    
    def from_quantum(self, state, hamiltonian):
        for node_id in state.nodes.keys():
            self.set_node(QNode(state[node_id], type="bra", connection="ham"))
            self.set_node(QNode(hamiltonian[node_id], type="ham"))
            self.set_node(QNode(state[node_id], type="ket", connection="ham"))

    def from_state(self, state, do_not_connect=[]):        
        for node_id in state.nodes.keys():
            if node_id in do_not_connect:
                self.set_node(QNode(state[node_id], type="bra"))
                self.set_node(QNode(state[node_id], type="ket"))
            else:
                self.set_node(QNode(state[node_id], type="bra", connection="ket"))
                self.set_node(QNode(state[node_id], type="ket", connection="bra"))
    
    def set_node(self, node: QNode):
        self.nodes[node.id] = node

    def find_largest_edge(self, do_not=()):
        tensor_max_shapes = dict([(node_id, max(self.nodes[node_id].tensor.shape)) for node_id in self.nodes.keys() if node_id not in do_not])
        node_id_max_shape = max(tensor_max_shapes, key=tensor_max_shapes.get)
        return node_id_max_shape
    
    def contract(self, holes=()):
        if type(holes)==str:
            holes = holes,
        node_names_reordered = list(holes)
        node_names_reordered = node_names_reordered + [node_id for node_id in self.nodes.keys() if node_id not in node_names_reordered]
        index_map = {v: i for i, v in enumerate(node_names_reordered)}
        self.nodes = dict(sorted(self.nodes.items(), key=lambda pair: index_map[pair[0]]))
        while len(self.nodes) > max(len(holes)+1, 1):
            node_a_key = self.find_largest_edge(do_not=holes)
            node_a = self.nodes.pop(node_a_key)
            connections = dict([(node_id, node_a.tensor.shape[node_a.connected_legs[node_id]]) for node_id in node_a.connected_legs.keys() if node_id not in holes and node_id in self.nodes.keys()])
            if len(connections)>0:
                node_b = self.nodes.pop(max(connections, key=connections.get))
            else:
                self.nodes[node_a_key] = node_a
                return self.nodes
            
            node_a_con_leg = node_a.connected_legs[node_b.id]
            node_b_con_leg = node_b.connected_legs[node_a.id]
            new_tensor = np.tensordot(node_a.tensor, node_b.tensor,
                                      axes=(node_a_con_leg, node_b_con_leg))
            new_id = node_a.id + node_b.id
            new_node = Node(new_id, new_tensor)

            for node in [node_a, node_b]:
                if len(node.contraction_history)==0:
                    new_node.contraction_history.append(node.id)
                else:
                    new_node.contraction_history += node.contraction_history

            self.rename_references(new_id, node_a, node_b)
            self.rename_references(new_id, node_b, node_a)

            new_node.connected_legs = node_a.connected_legs
            for node_id in new_node.connected_legs:
                if node_a_con_leg < new_node.connected_legs[node_id]:
                    new_node.connected_legs[node_id] += -1
            new_node.connected_legs.pop(node_b.id)
            for node_id in node_b.connected_legs.keys():
                if node_id != node_a.id:
                    leg_no = node_b.connected_legs[node_id]
                    if node_b_con_leg < leg_no:
                        leg_no += -1
                    leg_no += node_a.tensor.ndim - 1
                    if node_id not in new_node.connected_legs.keys():
                        new_node.connected_legs[node_id] = leg_no
                    else:
                        new_node.connected_legs[node_id] = (new_node.connected_legs[node_id], leg_no)
            
            new_node.group_parallel_legs()
            self.nodes[new_node.id] = new_node
        return self.nodes
    
    def rename_references(self, new_id, target_node, contraction_partner):
        for node_id in target_node.connected_legs:
            if node_id != contraction_partner.id and node_id in self.nodes.keys():
                keys = list(self.nodes[node_id].connected_legs.keys())
                if target_node.id in keys:
                    if new_id not in keys:
                        self.nodes[node_id].connected_legs[new_id] = self.nodes[node_id].connected_legs.pop(target_node.id)
                    else:
                        self.nodes[node_id].connected_legs[new_id] = (self.nodes[node_id].connected_legs[new_id], self.nodes[node_id].connected_legs.pop(target_node.id))
                        self.nodes[node_id].group_parallel_legs()


