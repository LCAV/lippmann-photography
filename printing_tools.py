"""
Created on 31.05.17

@author: miska
"""

import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
# this is used for plots 3D
from mpl_toolkits.mplot3d import Axes3D


def plot_pattern(arr, single=True):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if single:
        x0 = arr[:, 0]
        y0 = arr[:, 1]
        z0 = arr[:, 2]
        ax.plot(xs=x0, ys=y0, zs=z0, linewidth=1, alpha=0.5)
        ax.scatter(x0, y0, z0, marker="o", alpha=0.1)
    else:
        for arr_it in arr:
            x0 = arr_it[:, 0]
            y0 = arr_it[:, 1]
            z0 = arr_it[:, 2]
            ax.plot(xs=x0, ys=y0, zs=z0, linewidth=1, alpha=0.5)
            ax.scatter(x0, y0, z0, marker="o", alpha=0.1)
    plt.show()


def move_slow(p0, p_curr, s_max, ds, coord, margin):
    full = True
    if s_max - p_curr[coord] == 0:
        return p_curr, []

    if abs(s_max - p_curr[coord]) < 2 * margin:
        full = False
        margin = np.abs(s_max - p_curr[coord]) / 2

    arr = []
    direction = 1
    if p_curr[coord] - s_max > 0:
        direction = -1
    s_0 = p_curr[coord]
    a = ds ** 2 / (4 * margin)
    t_max = 2 * margin / ds
    p = list(p_curr)
    for t in np.arange(0, t_max, 1.0):
        s = a * t ** 2
        p[coord] = direction * s + s_0
        arr.append(list(p0 + p))

    # acelerate to marigin,
    if full:
        new_beg = list(p_curr)
        new_beg[coord] += direction * margin
        p, tmp_arr = move(p0, new_beg, s_max - direction * margin, ds, coord)
        arr += tmp_arr

    # descacelerate
    for t in np.arange(t_max, 0, -1.0):
        s = a * t ** 2
        p[coord] = -direction * s + s_max
        arr.append(list(p0 + p))
    return p, arr


def move(p0, p_curr, s_max, ds, coord):
    """Helper function for pattern2array3d, adds points between current point
     p_curr and p_curr shifted by s_max in direction coord
     (points are not further than ds, if the distance is to small does not add
     any points."""

    array = []
    p = p_curr
    if s_max < p_curr[coord]:
        ds = -ds
    for s in np.arange(p_curr[coord], s_max, ds):
        p[coord] = s
        array.append(list(p0 + p))

    return p, array

def wait(p0, p_curr, points):

    return p_curr, [list(p0 + p_curr)]*points


def pattern2array3d(rate, pattern, speed, z_steps, delta_z,
                    speed_limit=10, p0=np.array([0, 0, 0]),
                    y_range=100, z_range=100, point_limit=3333,

                    cut=False, margin=5):
    """ Create a list of positions at which the plate should be at the uniform
    time intervals (rate).

    rate        - rate at which the data is red: Range = 0.2 - 5, in milis
                it's the shortest time in which the plate can move
    pattern     - relative positions of dots (x=0 will be moved to p0)
    speed       - array of speeds with which to move along y direction
                (of the same length as pattern)
    z_steps     - number of layers in z direction
    delta_z     - distance between layers in z direction
    speed_limit - maximal speed of the piezo, should be 10, in microns/milis
    p0          - start position (position before first dot)
    y_range     - maximal value of y (not more than 100, in microns)
    z_range     - maximal value of z (not more than 100, in microns)
    cut         - if the array should have marks where to cut it
    margin      - at what distance to the end the piezo should slow down

    Returns the list of three-element lists (positions)"""

    if np.max(speed) > speed_limit:
        print("Maximal speed is bigger than speed limit: ", speed_limit, "\n")
        return

    dy = speed * rate
    p = [pattern[0], 0, 0]
    array = [np.copy(p0 + p)]
    ds = max(speed) * rate
    z_steps = min(z_steps, int(np.floor(z_range / delta_z)))
    odd_pattern = len(pattern) % 2 == 1
    dx = np.min(np.abs(pattern[:-1] - pattern[1:]))
    if dx == 0:
        dx = ds
    counter = 0

    for z in range(z_steps):
        for x in range(len(pattern)):
            if odd_pattern:
                inverted = (x + z) % 2 == 1
            else:
                inverted = x % 2 == 1
            p, tmp_arr = move(p0, p, pattern[x], ds=np.min([ds, 0.3 * dx]), coord=0)
            if cut and counter + len(tmp_arr) > point_limit:
                array.append([-1, -1, -1])
                counter = len(tmp_arr)
            else:
                counter += len(tmp_arr)
            array += list(tmp_arr)
            p[0] = pattern[x]
            p, tmp_arr = move_slow(p0, p, 0 if inverted else y_range, dy[x], coord=1, margin=margin)
            if cut and counter + len(tmp_arr) > point_limit:
                array.append([-1, -1, -1])
                counter = len(tmp_arr)
            else:
                counter += len(tmp_arr)
            array += list(tmp_arr)
        if z < z_steps - 1:
            # That's a hack! (0.5 ds)
            dz = np.min([ds, 0.3 * delta_z])
            p, tmp_arr = move(p0, p, -(z + 1) * delta_z, dz, coord=2)
            if cut and counter + len(tmp_arr) > point_limit:
                array.append([-1, -1, -1])
                counter = len(tmp_arr)
            else:
                counter += len(tmp_arr)
            array += list(tmp_arr)
            p[2] = -(z + 1) * delta_z
            pattern = pattern[::-1]
    return array


def pattern2array3d_pair(rate, pattern, speed, z_steps, delta_z,
                         speed_limit=10, p0=np.array([0, 0, 0]),
                         y_range=100, z_range=100, point_limit=3333, margin=5):
    return (pattern2array3d(rate, pattern, speed, z_steps, delta_z,
                            speed_limit, p0,
                            y_range, z_range, point_limit, cut=False, margin=margin),
            pattern2array3d(rate, pattern, speed, z_steps, delta_z,
                            speed_limit, p0,
                            y_range, z_range, point_limit,
                            cut=True, margin=margin))


def height_test(rate, line_len, z_steps,
                speed_limit=10.0, p0=np.array([0.0, 0.0, 100.0]), z_range=100.0, x_range=100.0, margin=5.0):
    """This is a specific pattern to print in order to check if the height 
    is correct"""

    p = [0, 0, 0]
    array = [np.copy(p0)]
    ds = speed_limit * rate
    delta_z = z_range / z_steps
    line_dist_x = (x_range - line_len) / z_steps

    for z in range(z_steps):
        y0 = p[1]
        x0 = p[0]
        p, tmp_arr = move_slow(p0, p, y0 + line_len, ds, 1, margin)
        array += list(tmp_arr)
        p, tmp_arr = wait(p0, p, 2)
        array += list(tmp_arr)
        p, tmp_arr = move_slow(p0, p, y0 + line_len * z / z_steps, ds, 1, margin)
        array += list(tmp_arr)
        p[1] = y0 + line_len * z / z_steps
        p, tmp_arr = move_slow(p0, p, x0 + line_len, ds, 0, margin)
        array += list(tmp_arr)
        p, tmp_arr = wait(p0, p, 2)
        array += list(tmp_arr)
        p, tmp_arr = move_slow(p0, p, x0, ds, 0, margin)
        array += list(tmp_arr)
        p[0] = x0
        p, tmp_arr = move_slow(p0, p, y0, ds, 1, margin)
        array += list(tmp_arr)
        p[1] = y0
        if z < z_steps - 1:
            p, tmp_arr = wait(p0, p, 2)
            array += list(tmp_arr)
            p, tmp_arr = move(p0, p, x0 + line_dist_x, ds, 0)
            array += list(tmp_arr)
            p[0] = x0 + line_dist_x
            p, tmp_arr = move(p0, p, -(z + 1) * delta_z, ds, 2)
            array += list(tmp_arr)
            p[2] = -(z + 1) * delta_z
    return array


def intensity2speed(t, dot_size, intensity, max_dots):
    intensity /= np.max(intensity)
    return dot_size / (intensity * t * max_dots)


def square_wave_array(p0, dx, nx, stepsx, dy, ny):
    """"Simple square wave in x and y directions
    which whe didn't managed to print """""
    assert (nx * ny < 3333)
    assert (dx * nx * stepsx < 100.0e3)
    assert (dy * ny < 100.0e3)
    array = []
    p = list(p0)
    for idx in range(nx):
        for _ in range(stepsx):
            p[0] += dx
            array.append(list(p))
        if idx % 2 == 0:
            for y in range(ny):
                p[1] += dy
                array.append(list(p))
        else:
            for y in range(ny):
                p[1] -= dy
                array.append((list(p)))
    return array


def circle_array(radius, n, p0):
    """"Simple circle, or rather regular polygon with n vertices"""""
    array = []
    for phi in np.linspace(0, 2 * np.pi, n):
        p = list(p0)
        p[0] += np.sin(phi) * radius
        p[1] += (1 + np.cos(phi)) * radius
        array.append(list(p))
    return array


def cross_array(max_shift, z_range=100, x_range=100, y_range=100,
                p0=np.array([50, 50, 0])):
    array = [np.copy(p0)]
    p = np.array([0, 0, 0])
    for z in np.arange(0, z_range, max_shift):
        for x in np.arange(0, 0.5 * x_range, max_shift):
            array.append(p0 + p)
            p[0] = x
        for x in np.arange(0.5 * x_range, -0.5 * x_range, -max_shift):
            array.append(p0 + p)
            p[0] = x
        for x in np.arange(-0.5 * x_range, 0, max_shift):
            array.append(p0 + p)
            p[0] = x
        for y in np.arange(0, 0.5 * y_range, max_shift):
            array.append(p0 + p)
            p[1] = y
        for y in np.arange(0.5 * y_range, -0.5 * y_range, -max_shift):
            array.append(p0 + p)
            p[1] = y
        for y in np.arange(-0.5 * y_range, 0, max_shift):
            array.append(p0 + p)
            p[1] = y
        p[2] = z
    return array


def array2file(time_period, pattern, filename,
               newline=";", delimiter=","):
    """"Writes array of points to the file in the format which can be then read
    by the program operating the piezo:

    time_period - time between instructions (0.2 ms tp 5ms)
    patter      - array (list) with points to write
                (each point is 3 element list)
    filename    - name of the .lipp (text) file
    newline     - marks end of line, should be ';'
    delimiter   - marks next number should be ','
    """""
    with open(filename, 'w') as file:
        print(time_period, newline, file=file)
        print(delimiter.join((map(str, pattern[0]))) + newline, file=file)
        for line in pattern:
            print(delimiter.join(map(str, line)) + newline, file=file)


def check_array(array, rate, max_speed=10, max_speed_change=10,
                max_points=3333, min_pos=0, max_pos=100):
    """""Checks if array is OK to print.

    Checks the number of points, the speed limit, the acceleration limit
    and if the coordinates are in the right cube

    array       - array (or list) of length-3 arrays (in microns)
    rate        - rate at which to move between points (in milliseconds)
    max_speed   - max. speed allowed by piezo (in microns over milliseconds)
    max_speed_change    - max. acceleration allowed by piezo (in microns over ms^2)
    max_points  - max. number of points which can be printed in one go
    min_pos     - min. position possible for the piezo
    max_pos     - max. position possible for the piezo
    """""
    array = list(array)
    checks_ok = True
    if len(array) > max_points:
        print("[Failed] To many points, expected:",
              max_points, "got:", len(array))
        checks_ok = False

    depth = 101

    array_ = np.concatenate(([array[0]], array, [array[-1]]))
    for (p1, p2, p3) in zip(array_[:-2], array_[1:-1], array_[2:]):
        if not hasattr(p2, "__len__") or len(p2) is not 3:
            print("[Failed] Wrong array format of the point:", p2)
            checks_ok = False
        if any(p2 < np.array(min_pos)):
            print("[Failed] Position smaller than", min_pos,
                  "for the point", p2)
            checks_ok = False
        if any(p2 > np.array(max_pos)):
            print("[Failed] Position bigger than", max_pos,
                  "for the point", p2)
            checks_ok = False
        s1 = (p2 - p1) / rate
        s2 = (p3 - p2) / rate
        if any(np.abs(s1) > np.array(max_speed)):
            print("[Failed] Speed", s1,
                  "bigger that", max_speed,
                  "for points", p1, "and", p2)
            checks_ok = False

        if any(np.abs(s2 - s1) > np.array(max_speed_change)):
            print("[Failed] Change in the speed", s2 - s1,
                  "bigger than", max_speed_change,
                  "for points", p1, p2, "and", p3)
            print("(rate", rate, ")")
            checks_ok = False

        if p2[2] > depth + 0.01:
            print("[Failed] path is going down around point: ", p2)
            checks_ok = False
        depth = p2[2]

    if checks_ok:
        print("[Success] All checks passed :)")
