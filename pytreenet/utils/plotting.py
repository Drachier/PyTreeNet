import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


class Coordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def plot_ttn(ttn, title=""):
    # TODO label legs
    
    Plot = PlotDetails(ttn)
    Plot.draw_children(ttn, ttn.root_id)
    
    _, ax = plt.subplots()

    # NODE NAMES
    for key in Plot.nodes.keys():
        node = Plot.nodes[key]
        if node.parent_coordinates is not None:
            ax.plot([node.parent_coordinates[0], node.coordinates[0]], [node.parent_coordinates[1], node.coordinates[1]], 'b', alpha=0.3)
        if len(ttn.nodes[key].open_legs) > 0:
            ax.plot([node.coordinates[0], node.coordinates[0]-.2], [node.coordinates[1], node.coordinates[1]+.07], 'b', alpha=0.7)
            index = ttn[key].open_legs
            plt.text(node.coordinates[0]-.2, node.coordinates[1]+.07, str(index)[1:-1], ha="right", color='black', size=10)
            for i, idx in enumerate(index):
                plt.text(node.coordinates[0]-.2, node.coordinates[1]-.13-.15*i, str(ttn[key].shape()[idx]), ha="right", color='blue', size=8)

    # LABEL
    patches = []
    for key in Plot.nodes.keys():
        circle = mpatches.Circle(Plot.nodes[key].coordinates, 0.1, ec="none")
        plt.text(Plot.nodes[key].coordinates[0]+0.2, Plot.nodes[key].coordinates[1]+0.15, key, ha="center", va="center", family='sans-serif', size=10, color='black')
        patches.append(circle)
    collection = PatchCollection(patches, color='blue', alpha=0.5)
    ax.add_collection(collection)

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.title(title)

    plt.show()


def plot_mps(ttn, title=""):    
    _, ax = plt.subplots()

    origin = Coordinates(0, 0)
    Plot = PlotDetails(ttn)
    Plot.draw_children(ttn, ttn.root_id)

    # LEGS
    for key in Plot.nodes.keys():
        node = Plot.nodes[key]
        if node.parent_coordinates is not None:
            ax.plot([origin.x+node.parent_coordinates[0], origin.x+node.coordinates[0]], [origin.y+node.parent_coordinates[1], origin.y+node.coordinates[1]], 'b', alpha=0.5)
        if len(ttn.nodes[key].open_legs) > 0:
            ax.plot([origin.x+node.coordinates[0], origin.x+node.coordinates[0]+.2], [origin.y+node.coordinates[1], origin.y+node.coordinates[1]], 'b', alpha=0.5)
            index = ttn[key].open_legs
            plt.text(origin.x+node.coordinates[0]+.2, origin.y+node.coordinates[1]-.1, str(ttn[key].shape()[index[0]]), ha="right", color='blue', size=8)

    # SITE
    patches = []
    for key in Plot.nodes.keys():
        circle = mpatches.Circle(Plot.nodes[key].coordinates, 0.1, ec="none")
        plt.text(origin.x+Plot.nodes[key].coordinates[0]-0.12, origin.y+Plot.nodes[key].coordinates[1], key, ha="right", va="center", family='sans-serif', size=10, color='black')
        patches.append(circle)
    collection = PatchCollection(patches, color='blue', alpha=0.5)
    ax.add_collection(collection)

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.title(title)

    plt.show()


# AUX for plotting

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
    def __init__(self, ttn):
        self.ttn = ttn
        self.nodes = dict()
        self.edges = []
        self.open_legs = []
        self.draw_root(ttn.root_id)
    
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