import numpy as np


def _get_normalized_distance_from_nearest_point(pixel_x: int,
                                                pixel_y: int,
                                                img_width: int,
                                                img_height: int,
                                                random_points: np.ndarray):
    shortest_norm_dist = 1
    points = (random_points-np.array([pixel_x, pixel_y]))/np.array([img_width/4, img_height/4])
    norm_points = np.linalg.norm(points, axis=1)
    shortest_norm_dist = np.min(norm_points) if np.min(norm_points) < shortest_norm_dist else shortest_norm_dist
    return shortest_norm_dist


def _map_to_bw_colour(colour_val: float):
    # We should return this as inverted so that pixels nearer to a selected
    # point is closer to a white colour.
    return (1 - colour_val) * 255


def Worley_noise(shape, num_points):
    img_width, img_height = shape[0], shape[1]
    img_grid = np.zeros(shape)
    # rand_points = _select_random_points(img_width, img_height, num_points)
    rand_x = np.random.randint(low=0, high=img_width, size=num_points)
    rand_y = np.random.randint(low=0, high=img_height, size=num_points)
    rand_points = np.array([rand_x, rand_y]).T

    for y in range(img_height):
        for x in range(img_width):
            bw_colour = _map_to_bw_colour(
                            _get_normalized_distance_from_nearest_point(
                                x, y, img_width, img_height, rand_points))
            # NOTE: Colour is in RGBA.
            img_grid[y, x] = np.round(bw_colour)
    worley = np.array(img_grid).astype('uint8')
    worley = worley/255.0
    return worley
