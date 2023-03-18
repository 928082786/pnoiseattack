import numpy as np
import utils
from opensimplex import OpenSimplex


def simplex_noise(shape, period):
    Sim = OpenSimplex()
    simplex = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            simplex[i, j] = Sim.noise2d(x=i/period, y=j/period)
    simplex = utils.normalize(simplex)
    return simplex
