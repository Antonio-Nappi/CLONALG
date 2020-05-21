import numpy as np
from numpy.random import uniform


def affinity(p_i, ap_rad, ap_cost, ds, aps, mst, P):
    """
    Description
    -----------
    Return the affinity of one subject.
    
    Parameters
    -----------
    p_i: numpy.array
        Subject of a population.
    
    Return
    -----------
    return: float
        Affinity of the subject passed as parameter.
    
    """

    print("aps")
    n_aps = 0
    for ap in aps:
        inside, _ = is_inside(p_i, 2 * ap_rad, ap)
        if inside:
            n_aps += 1

    print("clients")
    n_clients = 0
    sig = 0
    for client in ds:
        inside, distance = is_inside(p_i, ap_rad, client)
        if inside:
            n_clients += 1
            sig += P/(4*np.pi*distance)

    try:
        return ap_cost*n_aps + mst.sum() - n_clients - sig/n_clients
    except ZeroDivisionError:
        return ap_cost*n_aps + mst.sum() - n_clients


def is_inside(ap, rad, other):
    # Compare radius of circle with distance of its center from given point
    # print('ap0', ap[0], 'ap1', ap[1])
    # print('other0', other[0], 'other1', other[1])
    sq_dist = (other[0] - ap[0]) * (other[0] - ap[0]) + (other[1] - ap[1]) * (other[1] - ap[1])
    # print('sq', sq_dist)
    if sq_dist <= rad * rad:
        return (True, sq_dist)
    else:
        return (False, None)


def create_random_cells(population_size, problem_size, b_lo, b_up):
    population = [uniform(low=b_lo, high=b_up, size=problem_size) for x in range(population_size)]
    
    return population


def clone(p_i, clone_rate):
    clone_num = int(clone_rate / p_i[1])
    clones = [(p_i[0], p_i[1]) for x in range(clone_num)]
    
    return clones


def hypermutate(p_i, mutation_rate, b_lo, b_up):
    if uniform() <= p_i[1] / (mutation_rate * 100):
        ind_tmp = []
        for gen in p_i[0]:
            if uniform() <= p_i[1] / (mutation_rate * 100):
                ind_tmp.append(uniform(low=b_lo, high=b_up))
            else:
                ind_tmp.append(gen)
                
        return (np.array(ind_tmp), affinity(ind_tmp))
    else:
        return p_i


def select(pop, pop_clones, pop_size):
    population = pop + pop_clones
    
    population = sorted(population, key=lambda x: x[1])[:pop_size]
    
    return population


def replace(population, population_rand, population_size):
    population = population + population_rand
    population = sorted(population, key=lambda x: x[1])[:population_size]
    
    return population