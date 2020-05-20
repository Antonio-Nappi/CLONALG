import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

from clonalg_code import clonalg
from pprint import pprint

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
    plt.scatter(x1, y1, c="red")
    plt.scatter(x2, y2, c="blue")
    plt.show()

# Inputs parameters
b_lo, b_up = (-5, 5)

population_size = 100
selection_size = 10
problem_size = 2
random_cells_num = 20
clone_rate = 20
mutation_rate = 0.2
stop_condition = 1000

stop = 0

# Population <- CreateRandomCells(Population_size, Problem_size)
population = clonalg.create_random_cells(population_size, problem_size, b_lo, b_up)
best_affinity_it = []

while stop != stop_condition:
    # Affinity(p_i)
    population_affinity = [(p_i, clonalg.affinity(p_i)) for p_i in population]
    population_affinity = sorted(population_affinity, key=lambda x: x[1])

    best_affinity_it.append(population_affinity[:5])

    # Populatin_select <- Select(Population, Selection_size)
    population_select = population_affinity[:selection_size]

    # Population_clones <- clone(p_i, Clone_rate)
    population_clones = []
    for p_i in population_select:
        p_i_clones = clonalg.clone(p_i, clone_rate)
        population_clones += p_i_clones

    # Hypermutate and affinity
    pop_clones_tmp = []
    for p_i in population_clones:
        ind_tmp = clonalg.hypermutate(p_i, mutation_rate, b_lo, b_up)
        pop_clones_tmp.append(ind_tmp)
    population_clones = pop_clones_tmp
    del pop_clones_tmp

    # Population <- Select(Population, Population_clones, Population_size)
    population = clonalg.select(population_affinity, population_clones, population_size)
    # Population_rand <- CreateRandomCells(RandomCells_num)
    population_rand = clonalg.create_random_cells(random_cells_num, problem_size, b_lo, b_up)
    population_rand_affinity = [(p_i, clonalg.affinity(p_i)) for p_i in population_rand]
    population_rand_affinity = sorted(population_rand_affinity, key=lambda x: x[1])
    # Replace(Population, Population_rand)
    population = clonalg.replace(population_affinity, population_rand_affinity, population_size)
    population = [p_i[0] for p_i in population]

    stop += 1
    print(stop)

# We get the mean of the best 5 individuals returned by iteration of the above loop
bests_mean = []
iterations = [i for i in range(1000)]

for pop_it in best_affinity_it:
    bests_mean.append(np.mean([p_i[1] for p_i in pop_it]))

fig, ax = plt.subplots(1, 1, figsize = (5, 5), dpi=150)

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