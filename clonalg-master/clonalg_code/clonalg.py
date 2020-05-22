import numpy as np
from numpy.random import uniform

class Clonalg:
    def __init__(self,P=1,ap_cost=200,ap_rad=50,ds=None,aps=None,mst=None):
        self._P = P
        self._ap_cost = ap_cost
        self._ap_rad = ap_rad
        self._ds = ds
        self._aps = aps
        self._mst = mst

    def affinity(self,p_i):
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

        n_aps = 0
        for ap in self._aps:
            inside, _ = self.is_inside(p_i, 2 * self._ap_rad, ap)
            if inside:
                n_aps += 1

        n_clients = 0
        sig = 0
        for client in self._ds:
            inside, distance = self.is_inside(p_i, self._ap_rad, client)
            if inside:
                n_clients += 1
                sig += self._P/(4*np.pi*distance)

        if type(p_i) is tuple:
            index = [np.array_equal(p_i[0],x) for x in self._aps].index(True)

        else:
            index = [np.array_equal(p_i,x) for x in self._aps].index(True)

        row = self._mst[index,:]
        column = self._mst[:,index]
        mst_p_i = row+column.T
        try:
            return n_aps/len(self._aps) + mst_p_i.sum()/self._mst.sum() - n_clients/len(self._ds) - sig/n_clients
        except ZeroDivisionError:
            return n_aps/len(self._aps) + mst_p_i.sum()/self._mst.sum()


    def is_inside(self,ap, rad, other):
        if type(ap) is tuple:
            ap = ap[0]
        if type(other) is tuple:
            other = other[0]
        # Compare radius of circle with distance of its center from given point
        # print('ap0', ap[0], 'ap1', ap[1])
        # print('other0', other[0], 'other1', other[1])
        sq_dist = (other[0] - ap[0]) * (other[0] - ap[0]) + (other[1] - ap[1]) * (other[1] - ap[1])
        # print('sq', sq_dist)
        if sq_dist <= rad * rad:
            return (True, sq_dist)
        else:
            return (False, None)


    def create_random_cells(self,population_size, problem_size, b_lo, b_up):
        population = [uniform(low=b_lo, high=b_up, size=problem_size) for x in range(population_size)]

        return population


    def clone(self,p_i, clone_rate):
        clone_num = int(clone_rate / abs(p_i[1]))
        clones = [(p_i[0], p_i[1]) for x in range(clone_num)]

        return clones


    def hypermutate(self,p_i, mutation_rate, b_lo, b_up):
        if uniform() <= p_i[1] / (mutation_rate * 100):
            ind_tmp = []
            for gen in p_i[0]:
                if uniform() <= p_i[1] / (mutation_rate * 100):
                    ind_tmp.append(uniform(low=b_lo, high=b_up))
                else:
                    ind_tmp.append(gen)

            return np.array(ind_tmp)
        else:
            return p_i[0]


    def select(self,pop, pop_clones, pop_size):
        population = pop + pop_clones

        population = sorted(population, key=lambda x: x[1])[:pop_size]

        return population


    def replace(self,population, population_rand, population_size):
        population = population + population_rand
        population = sorted(population, key=lambda x: x[1])[:population_size]

        return population

    def set_aps(self,aps):
        self._aps = aps

    def set_mst(self,mst):
        self._mst = mst


