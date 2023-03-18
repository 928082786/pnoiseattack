import numpy as np


def voronoi_noise(grid_shape, img_shape):
    def seq(num_id):
        if num_id == 0:
            seq_x = [num_id, num_id + 1]
        elif num_id == img_shape[0] // grid_shape[0] - 1:
            seq_x = [num_id, num_id - 1]
        else:
            seq_x = [num_id - 1, num_id, num_id + 1]
        return seq_x

    def voronoi_distance(point):
        num_idx, num_idy = point[0] // grid_shape[0], point[1] // grid_shape[1]
        near = []
        for i in seq(num_idx):
            for j in seq(num_idy):
                near.append(random_points[i, j, :])
        near = np.array(near)
        norm = (point - near) / np.array([grid_shape[0] * 2, grid_shape[1] * 2])
        # shortest = min(np.min(np.linalg.norm(norm, axis=1)), shortest_distance)
        shortest = min(np.min(np.linalg.norm(norm, axis=1)), 1)
        return shortest

    # generate random_points
    random_points = np.zeros((img_shape[0] // grid_shape[0], img_shape[1] // grid_shape[1], 2))
    for i in range(0, img_shape[0], grid_shape[0]):
        for j in range(0, img_shape[1], grid_shape[1]):
            x = np.random.rand() * grid_shape[0] + i
            y = np.random.rand() * grid_shape[1] + j
            random_points[i // grid_shape[0], j // grid_shape[1], :] = np.array([x, y])

    voronio = np.zeros(img_shape)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            voronio[i, j] = voronoi_distance(np.array([i, j]))

    return voronio

