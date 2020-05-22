import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance
import time
# Dataset scattering
with open("Dataset 1 Wireless Access Point.txt", "r") as f1, open("Dataset 2 Wireless Access Point.txt", "r") as f2:
    x1, y1, x2, y2 = [], [], [], []
    for row in f1:
        r = re.split(" +", row)
        x1.append(float(r[1]))
        y1.append(float(r[2]))
    for row in f2:
        r = re.split(" +", row)
        x2.append(float(r[1]))
        y2.append(float(r[2]))
    # plt.scatter(x1, y1, c="red")
    # plt.show()
    # plt.scatter(x2, y2, c="blue")
    # plt.show()

# Dataset building
ds1 = np.vstack((x1, y1)).T
ds2 = np.vstack((x2, y2)).T

# Access points features
ap_rad = 50
ap_cost = 200
wire_unit_cost = 10
P = 1

# Inputs parameters
b_lo, b_up = (-500, 500)
population_size = 5
problem_size = 2
#hyperparametrs
selection_size = 2
random_cells_num = 3
clone_rate = 20
mutation_rate = 0.2

stop_condition = 100

stop = 0

def compute_mst(population,wire_unit_cost):
    graph = []
    for antibody in population:
        graph.append([wire_unit_cost * distance.euclidean(antibody, other) for other in population])
    graph = np.triu(graph)
    graph = csr_matrix(graph)
    mst = minimum_spanning_tree(graph).toarray()
    return mst