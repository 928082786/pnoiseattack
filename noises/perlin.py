import numpy as np
import noise

def normalize(img):
    min_ = np.min(img)
    max_ = np.max(img)
    n_img = (img-min_)/(max_-min_)
    return n_img


def perlin_noise(shape, period=4, octave=2, lacunarity=1):
    perlin = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            perlin[i, j] = noise.pnoise2(x=i/period, y=j/period, octaves=octave, lacunarity=lacunarity)
    perlin = normalize(perlin)
    return perlin

