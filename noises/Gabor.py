import numpy as np
import cv2


def valid_position(size, x, y):
    if x < 0 or x >= size: return False
    if y < 0 or y >= size: return False
    return True

# Procedural Noise
# Note: Do not take these as optimized implementations.

'''
Gabor kernel

sigma       variance of gaussian envelope
theta         orientation
lambd       sinusoid wavelength, bandwidth
xy_ratio    value of x/y
psi            phase shift of cosine in kernel
sides        number of directions
'''


def gaborK(ksize, sigma, theta, lambd, xy_ratio, sides):
    gabor_kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, xy_ratio, 0, ktype=cv2.CV_32F)
    for i in range(1, sides):
        gabor_kern += cv2.getGaborKernel((ksize, ksize), sigma, theta + np.pi * i / sides, lambd, xy_ratio, 0,
                                         ktype=cv2.CV_32F)
    return gabor_kern


'''
Gabor noise
- randomly distributed kernels
- anisotropic when sides = 1, pseudo-isotropic for larger "sides"
'''


def gaborN_rand(size, grid, num_kern, ksize, sigma, theta, lambd, xy_ratio=1, sides=1, seed=0):
    np.random.seed(seed)

    # Gabor kernel
    if sides != 1:
        gabor_kern = gaborK(ksize, sigma, theta, lambd, xy_ratio, sides)
    else:
        gabor_kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, xy_ratio, 0, ktype=cv2.CV_32F)

    # Sparse convolution noise
    sp_conv = np.zeros([size, size])
    dim = int(size / 2 // grid)
    noise = []
    for i in range(-dim, dim + 1):
        for j in range(-dim, dim + 1):
            x = i * grid + size / 2 - grid / 2
            y = j * grid + size / 2 - grid / 2
            for _ in range(num_kern):
                dx = np.random.randint(0, grid)
                dy = np.random.randint(0, grid)
                while not valid_position(size, x + dx, y + dy):
                    dx = np.random.randint(0, grid)
                    dy = np.random.randint(0, grid)
                weight = np.random.random() * 2 - 1
                sp_conv[int(x + dx)][int(y + dy)] = weight

    sp_conv = cv2.filter2D(sp_conv, -1, gabor_kern)
    return sp_conv


'''
Gabor noise
- controlled, uniformly distributed kernels

grid        ideally is odd and a factor of size
thetas    orientation of kernels, has length (size / grid)^2
'''


def gaborN_uni(size, grid, ksize, sigma, lambd, xy_ratio, thetas):
    sp_conv = np.zeros([size, size])
    temp_conv = np.zeros([size, size])
    dim = int(size / 2 // grid)

    for i in range(-dim, dim + 1):
        for j in range(-dim, dim + 1):
            x = i * grid + size // 2
            y = j * grid + size // 2
            temp_conv[x][y] = 1
            theta = thetas[(i + dim) * dim * 2 + (j + dim)]

            # Gabor kernel
            gabor_kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, xy_ratio, 0, ktype=cv2.CV_32F)
            sp_conv += cv2.filter2D(temp_conv, -1, gabor_kern)
            temp_conv[x][y] = 0

    return sp_conv
