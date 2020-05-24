import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance
import time
import csv
from clonalg_code.clonalg import Clonalg
import csv
# Set the seed
np.random.seed(1234)
seeds = [np.random.randint(9999) for i in range(100)]

# Open the dataset
with open("../Dataset 1 Wireless Access Point.txt", "r") as f1, open("../Dataset 2 Wireless Access Point.txt",
                                                                     "r") as f2:
    x1, y1, x2, y2 = [], [], [], []
    for row in f1:
        r = re.split(" +", row)
        x1.append(float(r[1]))
        y1.append(float(r[2]))
    for row in f2:
        r = re.split(" +", row)
        x2.append(float(r[1]))
        y2.append(float(r[2]))

    # Scatter the dataset
    # plt.scatter(x1, y1, c="red")
    # plt.show()
    # plt.scatter(x2, y2, c="blue")
    # plt.show()

# Build the dataset
ds1 = np.vstack((x1, y1)).T
ds2 = np.vstack((x2, y2)).T

with open('seeds.csv','r') as seedsfile:
    line = seedsfile.readline()
    seeds = line.split(',')
    seeds = [int(seed) for seed in seeds]
'''
for seed in seeds:
    np.random.seed(seed)
    parameters = {'population_size':[64],
                  'random_cells_factor':[0.4],
                  'selection_size_factor':[0.5],
                  'clone_rate':[100],
                  'mutation_rate':[0.25],
                  'stop_condition':[50]}
                  '''
# Access points features
ap_rad = 50  # Access point radius
ap_cost = 200  # Access point cost
wire_unit_cost = 10  # Wire unit cost
P = 1  # Access point signal power
k_cost = 100  # General cost penalty
k_n_client = 0.5  # Client cost bonus

# Inputs parameters
b_lo, b_up = (-500, 500)  # Dimension limits
population_size = 12  # Number of antibodies (i.e. APs)
problem_size = 2  # Space dimensionality

# Hyperparametrs
selection_size = 4  # Selection size
random_cells_num = 7  # Number of random antibodies
clone_rate = 100  # Clone rate
mutation_rate = 0.8  # Mutation rate
stop_condition = 100  # Number of iterations


# Function for Minimum Spanning Tree
def compute_mst(population, wire_unit_cost):
    graph = []

    for antibody in population:
        graph.append([wire_unit_cost * distance.euclidean(antibody, other) for other in population])

    graph = np.triu(graph)
    graph = csr_matrix(graph)
    mst = minimum_spanning_tree(graph).toarray()

    return mst


# Create CLONALG object
cln = Clonalg(P=P, ap_cost=ap_cost, ap_rad=ap_rad, ds=ds1, k_cost=k_cost, k_n_client=k_n_client)

# Create source radio station and initial random population
population = [np.array([-400, 300])]  # Source radio station
population.extend(cln.create_random_cells(population_size - 1, problem_size, b_lo, b_up))
# print("Initial population", population)

# Set population and MST in CLONALG object
cln.set_aps(population)
cln.set_mst(compute_mst(population, wire_unit_cost))

# Initialize list of best affinities
best_affinity_it = []

# Initialize time counter
p_time = time.time()

# Start iterations
stop = 0

while stop != stop_condition:
    print("Iteration", stop)

    # - Compute affinity for each antibody of population
    # - Sort by best affinity
    # - Save affinities
    population_affinity = [(p_i, cln.affinity(p_i)) for p_i in population]
    # print("Original Population affinity", population_affinity)
    population_affinity[1:] = sorted(population_affinity[1:], key=lambda x: x[1])
    best_affinity_it.append(population_affinity)

    # Select the best subset from population
    population_select = population_affinity[1:selection_size + 1]

    # Create a number of clones of antibodies from the selected subset (higher affinity, more clones)
    population_clones = []
    for p_i in population_select:
        p_i_clones = cln.clone(p_i, clone_rate)
        population_clones += p_i_clones
        print("Population clones", population_clones)
        print("Length population clones", len(population_clones))

        # Hypermutate clones
        pop_clones_tmp = [population[0]]
        for p_i in population_clones:
            ind_tmp = cln.hypermutate(p_i, mutation_rate, b_lo, b_up)
            pop_clones_tmp.append(ind_tmp)
        # print("Population of mutated clones", pop_clones_tmp)
        # print("Length population of mutated clones", len(pop_clones_tmp))

        # Compute affinity for each antibody of clones population
        cln.set_aps(pop_clones_tmp)
        cln.set_mst(compute_mst(pop_clones_tmp, wire_unit_cost))
        population_clones_affinity = [(p_i, cln.affinity(p_i)) for p_i in pop_clones_tmp]
        # print("Population clones with affinity", population_clones)
        # print("Length population clones with affinity", len(population_clones))
        del pop_clones_tmp

        # Select a set of best antibodies between initial population and clones with original size
        population_org_clones_affinity = [population_affinity[0]]
        population_org_clones_affinity.extend(
            cln.select(population_affinity[1:], population_clones_affinity, population_size - 1))

        # Create random antibodies
        population_rand = [population[0]]
        population_rand.extend(cln.create_random_cells(random_cells_num - 1, problem_size, b_lo, b_up))
        cln.set_aps(population_rand)
        cln.set_mst(compute_mst(population_rand, wire_unit_cost))
        population_rand_affinity = [(p_i, cln.affinity(p_i)) for p_i in population_rand]
        population_rand_affinity[1:] = sorted(population_rand_affinity[1:], key=lambda x: x[1])

        # Select a set of best antibodies between population with clones and random antibodies with original size
        population[1:] = cln.replace(population_org_clones_affinity[1:], population_rand_affinity[1:],
                                     population_size - 1)
        population[1:] = [p_i[0] for p_i in population[1:]]
        # print("Final population", population)
        cln.set_aps(population)
        cln.set_mst(compute_mst(population, wire_unit_cost))

    # print("End iteration", stop)
    stop += 1

e_time = time.time() - p_time
print("Elapsed time:", e_time / 60, "minutes")

# - Scatter clients and APs
# - Draw APs MST
plt.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.2)
plt.xticks(np.arange(-500, 500, 100))
plt.yticks(np.arange(-500, 500, 100))
plt.scatter(x1, y1, c="red")
# plt.scatter(population[0][0], population[0][1], c="green")

fig = plt.gcf()
ax = fig.gca()
ax.add_artist(plt.Rectangle((population[0][0], population[0][1]), 40, 40, facecolor='green', edgecolor='green'))
for p_i in population[1:]:
    ax.add_artist(plt.Circle((p_i[0], p_i[1]), ap_rad, facecolor='none', edgecolor='blue'))
mst = compute_mst(population, wire_unit_cost)
for i in range(len(mst)):
    for j in range(len(mst[i])):
        if mst[i][j] != 0:
            plt.plot([population[i][0], population[j][0]], [population[i][1], population[j][1]], c="blue")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Get the mean of the affinities returned by iteration of the above loop
bests_mean = []
iterations = [i for i in range(stop_condition)]

for pop_it in best_affinity_it:
    bests_mean.append(np.mean([p_i[1] for p_i in pop_it]))
# print('bests', len(bests_mean))
# print(bests_mean)
fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
# print(iterations)
# sns.set_style("darkgrid")
# sns.pointplot(x=iterations, y=bests_mean)
#
# plt.tick_params(
#     axis='x',  # changes apply to the x-axis
#     which='both',  # both major and minor ticks are affected
#     bottom=False,  # ticks along the bottom edge are off
#     top=False,  # ticks along the top edge are off
#     labelbottom=True)  # labels along the bottom edge are on

plt.xticks(np.arange(0, stop_condition, stop_condition / 10))
plt.ylim(-10, 50)
plt.plot(iterations, bests_mean)
plt.title("Mean of Affinities by Iteration", fontsize=12)
plt.ylabel("Affinity Mean", fontsize=10)
plt.rc('ytick', labelsize=2)
plt.xlabel("# Iteration", fontsize=10)
plt.show()
