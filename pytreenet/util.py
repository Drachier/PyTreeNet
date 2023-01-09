"""
Some useful tools
"""
import numpy as np
import matplotlib as plt

from copy import deepcopy

def crandn(size):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    # 1/sqrt(2) is a normalization factor        
    return (np.random.standard_normal(size)
        + 1j*np.random.standard_normal(size)) / np.sqrt(2)


def crandn_uniform(size):
    """
    Draw random samples from a uniform [-1, 1] distribution.
    """
    # 1/sqrt(2) is a normalization factor        
    return (np.random.uniform(low=-1, high=1, size=size)
        + 1j*np.random.uniform(low=-1, high=1, size=size)) / np.sqrt(2)


def pauli_matrices(asarray=True):
    """
    Returns the three Pauli matrices X, Y, and Z in Z-basis as ndarray, if asarray is True
    otherwise it returns them as lists.
    """
    X = [[0,1],
         [1,0]]
    Y = [[0,-1j],
         [1j,0]]
    Z = [[1,0],
         [0,-1]]
    if asarray:
        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

    return X, Y, Z

def copy_object(obj, deep):
    """
    Returns a normal copy of obj, if deep=False and a deepcopy if deep=True.
    """
    if deep:
        new_obj = deepcopy(obj)
    else:
        new_obj = obj
        
    return new_obj


#
# Plotting Utils
# TODO comment stuff


def xy_from_polar(polar):
    polar[1] = polar[1] / 360 * 2 * np.pi
    return np.array([polar[0]*np.cos(polar[1]), polar[0]*np.sin(polar[1])]) 


class NodePlot:
    def __init__(self, ttn, key, coordinates, parent_coordinates, angle_area):
        self.ttn = ttn
        self.key = key
        self.coordinates = coordinates
        self.parent_coordinates = parent_coordinates
        self.angle_area = angle_area
        

class PlotDetails:
    def __init__(self, ttn, root_node_key):
        self.ttn = ttn
        self.nodes = dict()
        self.edges = []
        self.open_legs = []
        self.draw_root(root_node_key)
    
    def draw_children(self, ttn, parent_node_key):
        Parent = self.nodes[parent_node_key]
        children_keys = ttn.nodes[parent_node_key].children_legs.keys()

        number_of_children = len(children_keys)
        segment = 1
        for child_key in children_keys:
            segment_length = (Parent.angle_area[1] - Parent.angle_area[0]) / number_of_children
            segment_start = Parent.angle_area[0] + (segment-1) * segment_length 
            segment_end = segment_start + segment_length

            self.nodes[child_key] = NodePlot(self.ttn, child_key, Parent.coordinates + xy_from_polar([1, segment_start + segment_length / 2]), Parent.coordinates, [segment_start, segment_end])
            self.draw_children(ttn, child_key)

            segment += 1
        return None

    def draw_root(self, root_node_key):
        self.nodes[root_node_key] = NodePlot(self.ttn, root_node_key, np.array([0, 0]), None, [180, 360])
        return None


