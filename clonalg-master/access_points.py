import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance

from clonalg_code.clonalg import Clonalg
from pprint import pprint

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
population_size = 50
problem_size = 2

selection_size = 1
random_cells_num = 20
clone_rate = 20
mutation_rate = 0.2

stop_condition = 100

stop = 0
cln = Clonalg(P,ap_cost,ap_rad,ds1)
# Population <- CreateRandomCells(Population_size, Problem_size)
population = cln.create_random_cells(population_size, problem_size, b_lo, b_up)

# Graph and MST
graph = []

for antibody in population:
    graph.append([wire_unit_cost*distance.euclidean(antibody,other) for other in population])

graph = np.triu(graph)
graph = csr_matrix(graph)
mst = minimum_spanning_tree(graph).toarray()
cln.set_aps(population)
cln.set_mst(mst)
best_affinity_it = []

while stop != stop_condition:
    # Affinity(p_i)

    population_affinity = [(p_i, cln.affinity(p_i)) for i, p_i in enumerate(population)]
    population_affinity = sorted(population_affinity, key=lambda x: x[1])

    best_affinity_it.append(population_affinity[:5])

    # Populatin_select <- Select(Population, Selection_size)
    population_select = population_affinity[:selection_size]

    # Population_clones <- clone(p_i, Clone_rate)
    population_clones = []
    for p_i in population_select:
        p_i_clones = cln.clone(p_i, clone_rate)
        population_clones += p_i_clones

    # Hypermutate and affinity
    pop_clones_tmp = []
    for p_i in population_clones:
        ind_tmp = cln.hypermutate(p_i, mutation_rate, b_lo, b_up)
        pop_clones_tmp.append(ind_tmp)
    population_clones = pop_clones_tmp
    del pop_clones_tmp

    # Population <- Select(Population, Population_clones, Population_size)
    population = cln.select(population_affinity, population_clones, population_size)
    cln.set_aps(population)
    # Population_rand <- CreateRandomCells(RandomCells_num)
    population_rand = cln.create_random_cells(random_cells_num, problem_size, b_lo, b_up)
    cln.set_aps(population_rand)
    population_rand_affinity = [(p_i, cln.affinity(p_i)) for i, p_i in enumerate(population_rand)]
    population_rand_affinity = sorted(population_rand_affinity, key=lambda x: x[1])
    # Replace(Population, Population_rand)
    population = cln.replace(population_affinity, population_rand_affinity, population_size)
    population = [p_i[0] for p_i in population]
    cln.set_aps(population)
    stop += 1
    print(stop)

plt.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.2)
plt.xticks(np.arange(-500, 500, 50))
plt.yticks(np.arange(-500, 500, 50))
plt.scatter(x1, y1, c="red")
fig = plt.gcf()
ax = fig.gca()
for p_i in population:
    ax.add_artist(plt.Circle((p_i[0], p_i[1]), ap_rad, facecolor='none', edgecolor='blue'))
#plt.scatter([p_i[0] for p_i in population], [p_i[1] for p_i in population], s=ap_rad*ap_rad*np.pi/4, facecolors='none', edgecolors='blue')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# We get the mean of the best 5 individuals returned by iteration of the above loop
bests_mean = []
iterations = [i for i in range(stop_condition)]

for pop_it in best_affinity_it:
    bests_mean.append(np.mean([p_i[1] for p_i in pop_it]))
print('bests',len(bests_mean))
print(bests_mean)
fig, ax = plt.subplots(1, 1, figsize = (5, 5), dpi=150)
print(iterations)
sns.set_style("darkgrid")
sns.pointplot(x=iterations, y=bests_mean)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title("Mean of 5 Best Individuals by Iteration", fontsize=12)
plt.ylabel("Affinity Mean", fontsize=10)
plt.rc('ytick',labelsize=2)
plt.xlabel("# Iteration", fontsize=10)
plt.show()