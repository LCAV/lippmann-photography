"""
Created on 31.05.17

@author: miska
"""
import math
import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
# this is used for plots 3D
from mpl_toolkits.mplot3d import Axes3D

""" Naming conventions in this file (should we use classes or namespaces?)
    _pattern - returns positions of the dots
    _array   - returns positions 3D at equal timestamps

    all 3D arrays should be in the format Nx3, because that's the format
    we print later"""


class Trajectory:
    """"Abstract class which represent trajectory to be approximated using
    piecewise straight line, with knots at grid points"""""

    def plot(self, res=100):
        """"Returns tree arrays - of x, y, and z which can be plotted with
        matplotlib"""""
        raise NotImplementedError

    def position(self, t):
        """"Returns position at time t"""""
        raise NotImplementedError

    def distance_from_segment(self, t_min, t_max, a, b):
        """"Returns the distance between part of the curve
        defined by t_min and t_max and a linear segment (a,b)"""""
        raise NotImplementedError

    def block_position(self, t, space_resolution, space_range, i3D):
        """"This is helper function for self.approximate
        which returns position of i3D neighbour of the curve at time t
        (where i3D is a 3D index)"""""
        nbh = (np.array(range(space_range)) - np.floor(space_range / 2)) \
            * space_resolution
        p = np.floor(self.position(t) / space_resolution) \
            * space_resolution
        return p + np.array([nbh[i3D[0]], nbh[i3D[1]], nbh[i3D[2]]])

    @staticmethod
    def two_segments_distance(p0, p1, a, b):
        A = p0 - a
        B = p1 - b
        return (A.dot(A) + A.dot(B) + B.dot(B)) / 3.0

    def approximate(self, space_resolution, segments, space_range=3):
        """""Calculates the best approximation to the curve (trajectory)
        assuming that knots are at uniform timestamps, and searching
        only the space of close paths, and not all possible paths

        space_resolution    - the resolution of the grid
        segments            - number of points to use in the approximation
        space_range         - number of points in the neighbourhood
        (in every direction) considered at each step of the search

        Returns error and np.array with positions along the approximation
        (a[0,:] are x coordinates and so on)
        """""
        time_grid = np.linspace(0, 1, segments)
        dimensions = [space_range] * 3
        errors = np.zeros(dimensions)
        paths = np.empty(dimensions, dtype=list)

        positions = [self.position(t) for t in time_grid]
        i3D = []
        for xi in range(space_range):
            for yi in range(space_range):
                for zi in range(space_range):
                    i3D.append((xi, yi, zi))
                    paths[xi, yi, zi] = list([positions[0]])

        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            new_paths = paths
            new_errors = errors
            for i in i3D:
                p1 = self.block_position(t1, space_resolution, space_range, i)
                min_error = math.inf
                best_j = (0, 0, 0)
                for j in i3D:
                    err = errors[j]
                    p0 = np.copy(paths[j][-1])
                    err += self.distance_from_segment(t0, t1, p0, p1)
                    if err < min_error:
                        min_error = err
                        best_j = j
                new_errors[i] = min_error
                new_paths[i] = list(paths[best_j])
                new_paths[i].append(p1)
            paths = new_paths
            errors = new_errors

        min_error = math.inf
        best_i = (0, 0, 0)

        for i in i3D:
            if errors[i] < min_error:
                min_error = errors[i]
                best_i = i
        return min_error, np.array(paths[best_i])


class StraightLine(Trajectory):
    """"Straight line from point p0 to point p1"""""

    def __init__(self, p0, p1):
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)

    def position(self, t):
        return (1 - t) * self.p0 + t * self.p1

    def distance_from_segment(self, t_min, t_max, a, b):
        p0 = self.position(t_min)
        p1 = self.position(t_max)
        self.two_segments_distance(p0, p1, a, b)

    def plot(self, res=100):
        xs = np.linspace(self.p0[0], self.p1[0], res)
        ys = np.linspace(self.p0[1], self.p1[1], res)
        zs = np.linspace(self.p0[2], self.p1[2], res)
        return xs, ys, zs


class NicelyParameterized(Trajectory):
    """"Trajectory given by the set points at uniform timestamps"""""

    def __init__(self, xs, ys, zs):
        self.pos = np.zeros((3, len(xs)))
        self.pos[:, 0] = xs
        self.pos[:, 1] = ys
        self.pos[:, 2] = zs
        self.dt = 1.0 / (len(xs) - 1)

    def position(self, t):
        i = math.floor(t / self.dt)
        if i >= len(self.pos[:, 0]) - 1:
            return self.pos[-1, :]
        t1 = t / self.dt - i
        return np.array(self.pos[i, :] * (1 - t1) + self.pos[i + 1, :] * t1)

    def distance_from_segment(self, t_min, t_max, a, b):
        i_min = math.floor(t_min / self.dt)
        i_max = math.floor(t_max / self.dt) + 1
        times = np.arange(i_min, i_max) * self.dt
        times[0] = t_min
        times[-1] = t_max
        dist = 0
        for t1, t2 in zip(times[:-1], times[1:]):
            p0 = self.position(t1)
            p1 = self.position(t2)
            dist += self.two_segments_distance(
                p0,
                p1,
                (1 - t1) * a + t1 * b,
                (1 - t2) * a + t2 * b)
        return dist

    def plot(self, res=100):
        return self.pos[:, 0], self.pos[:, 0], self.pos[:, 0]

    def limit_speed(self, speed, rate, iterations=100, beta=0.1):
        positions = np.copy(self.pos)
        for _ in range(iterations):
            v = (positions[1:, :] - positions[:-1, :]) / rate
            sign_v = np.sign(v)
            abs_v = np.array(np.abs(v) - speed)
            if (abs_v < speed).all():
                break
            abs_v = (abs_v < 0).choose(abs_v, 0)
            v = abs_v * sign_v
            positions[1:, :] -= beta * v
            positions[:-1, :] += beta * v

        diff = (self.pos - positions)
        err = np.diag(diff.dot(diff.T))
        err = np.mean(err)
        return err, positions


def pattern2array1D(rate, max_dots, T, p0, pattern, intensity, reverse=False):
    """" Create a list of positions at which the plate should be at the uniform
    time intervals (rate). This is approximation that may print off up to one
    dot per stack of dots (but his should not matter if number of dots printed
    in one place is of order of 50).

    rate        - rate at which the data is red: Range = 0.2ms - 5ms,
                it's the shortest time in which the plate can move
    max_dots    - maximal number of dots that can be printed in one place such
                    that the index of refraction stays linear
    T           - time between two laser pulses
    p0          - start position (position before first dot)
    pattern     - relative positions of dots (x=0 will be moved to p)
    intensity   - ratio of the intensity of the dot to the max. intensity

    Returns the list of three-element lists (positions)"""""

    print("number of entries read between pulses:", rate / T)
    max_time = T * max_dots
    max_repeat = max_time / rate
    array = []
    if reverse:
        pattern = pattern[::-1]
        intensity = intensity[::-1]
    for x, i in zip(pattern, intensity):
        p = list(p0)
        p[0] += x
        for _ in range(math.floor(i * max_repeat)):
            array.append(p)
    return array


def pattern2array3D(rate, max_dots, T, pattern, intensity, dot_size,
                    z_range, y_range, p0=np.array([0, 0, 0]), dx=5, dz=5):
    """" Create a list of positions at which the plate should be at the uniform
    time intervals (rate). This is approximation that may print off up to one
    dot per stack of dots (but his should not matter if number of dots printed
    in one place is of order of 50).

    rate        - rate at which the data is red: Range = 0.2ms - 5ms,
                it's the shortest time in which the plate can move
    max_dots    - maximal number of dots that can be printed in one place such
                    that the index of refraction stays linear
    T           - time between two laser pulses
    p0          - start position (position before first dot)
    pattern     - relative positions of dots (x=0 will be moved to p)
    intensity   - ratio of the intensity of the dot to the max. intensity
    dot_size    - vector of dot sizes (in x, y and z)

    Returns the list of three-element lists (positions)"""""

    # one might need to add step size in x
    dy = dot_size[1] * rate / T / max_dots / intensity
    pattern_ = pattern
    p = [pattern[0], 0, 0]
    array = [np.copy(p0+p)]
    z_steps = math.floor(z_range / dot_size[2])
    x_steps = len(pattern)
    for z in range(z_steps):
        for x in range(x_steps):
            print("x:", x)
            # if the current layer is forwards or backwards
            p[1] = 0
            direction = 1
            if (x_steps % 2 == 1) and ((x + z) % 2 == 1):
                p[1] = y_range
                direction = -1
            if (x_steps % 2 == 0)and ( x % 2 == 1):
                p[1] = y_range
                direction = -1
            sign = np.sign(pattern_[x]-p[0])
            for x_step in np.arange(p[0], pattern_[x], sign * dx):
                p[0] = float(x_step)
                # step in x
                array.append(list(p0+p))
            p[0] = pattern_[x]
            array.append(list(p0+p))
            for y in range(math.floor(y_range / dy[x])):
                p[1] += direction * dy[x]
                # step in y
                array.append(list(p0+p))
            curr_z = p[2]
        if z < z_steps - 1:
            for z_change in np.arange(curr_z, curr_z + dot_size[2], dz):
                # print(z_change)
                p[2] = z_change
                array.append(list(p0+p))
            p[2] = curr_z + dot_size[2]
            pattern_ = pattern_[::-1]
    return array


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

def cross_array(max_shift, z_range=100, x_range=100, y_range=100, p0 = np.array([50, 50, 0])):
    array=[np.copy(p0)]
    p = np.array([0, 0, 0])
    for z in np.arange(0, z_range, max_shift):
        for x in np.arange(0, 0.5*x_range, max_shift):
            array.append(p0+p)
            p[0] = x
        for x in np.arange(0.5*x_range, -0.5*x_range, -max_shift):
            array.append(p0+p)
            p[0] = x
        for x in np.arange(-0.5*x_range, 0, max_shift):
            array.append(p0+p)
            p[0] = x
        for y in np.arange(0, 0.5*y_range, max_shift):
            array.append(p0+p)
            p[1] = y
        for y in np.arange(0.5*y_range, -0.5*y_range, -max_shift):
            array.append(p0+p)
            p[1] = y
        for y in np.arange(-0.5*y_range, 0, max_shift):
            array.append(p0+p)
            p[1] = y
        p[2] = z
    return  array



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


def check_array(array, rate, max_speed=10, max_speed_change=100,
                max_points=3333, min_pos=0, max_pos=100):
    """""Checks if array is OK to print.

    Checks the number of points, the speed limit, the acceleration limit
    and if the coordinates are in the right cube

    array       - array (or list) of length-3 arrays (in microns)
    rate        - rate at which to move between points (in milliseconds)
    max_speed   - max. speed allowed by piezo (in microns over milliseconds)
    max_acc     - max. acceleration allowed by piezo (in microns over ms^2)
    max_points  - max. number of points which can be printed in one go
    min_pos     - min. position possible for the piezo
    max_pos     - max. position possible for the piezo
    """""
    array = list(array)
    checks_OK = True
    if len(array) > max_points:
        print("[Failed] To many points, expected:",
              max_points, "got:", len(array))
        checks_OK = False

    array_ = np.concatenate(([array[0]], array, [array[-1]]))
    for (p1, p2, p3) in zip(array_[:-2], array_[1:-1], array_[2:]):
        if not hasattr(p2, "__len__") or len(p2) is not 3:
            print("[Failed] Wrong array format of the point:", p2)
            checks_OK = False
        if any(p2 < np.array(min_pos)):
            print("[Failed] Position smaller than", min_pos,
                  "for the point", p2)
            checks_OK = False
        if any(p2 > np.array(max_pos)):
            print("[Failed] Position bigger than", max_pos,
                  "for the point", p2)
            checks_OK = False
        s1 = (p2 - p1) / rate
        s2 = (p3 - p2) / rate
        if any(np.abs(s1) > np.array(max_speed)):
            print("[Failed] Speed", s1,
                  "bigger that", max_speed,
                  "for points", p1, "and", p2)
            checks_OK = False

        if any(np.abs(s2 - s1) > np.array(max_speed_change)):
            print("[Failed] Change in the speed", s2 - s1,
                  "bigger than", max_speed_change,
                  "for points", p1, p2, "and", p3)
            print("(rate", rate, ")")
    if checks_OK:
        print("[Success] All checks passed :)")


# noinspection PyTypeChecker
def approximation_example():
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 200)
    z = np.linspace(0, 1, 200)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    some_line = NicelyParameterized(x, y, z)
    # some_line = StraightLine([0,0,0],[1,2,3])
    error, path = some_line.approximate(0.1, 10, space_range=3)
    print(error)
    print("----")
    x1 = path[0, :]
    y1 = path[1, :]
    z1 = path[2, :]

    ox, oy, oz = some_line.plot()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xs=ox, ys=oy, zs=oz)
    ax.plot(xs=x1, ys=y1, zs=z1)
    plt.show()


# noinspection PyTypeChecker
def speed_limit_example():
    theta = np.linspace(0, 4 * np.pi, 400)
    z = np.linspace(0, 1, 400)
    r = z ** 2 + 1
    x = r * np.sin(theta + 0.3 * theta ** 2)
    y = r * np.cos(theta + 0.3 * theta ** 2)

    some_line = NicelyParameterized(x, y, z)
    # some_line = StraightLine([0,0,0],[1,2,3])
    error, positions = some_line.limit_speed(speed=0.2, rate=1)
    print(error)
    print("----")
    x1 = positions[0, :]
    y1 = positions[1, :]
    z1 = positions[2, :]

    some_line = NicelyParameterized(x1, y1, z1)
    error, path = some_line.approximate(0.01, 200)
    print(error)
    print("----")
    x2 = path[0, :]
    y2 = path[1, :]
    z2 = path[2, :]

    some_line = NicelyParameterized(x2, y2, z2)
    error, positions = some_line.limit_speed(speed=0.2, rate=2)
    print(error)
    print("----")
    x3 = positions[0, :]
    y3 = positions[1, :]
    z3 = positions[2, :]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xs=x, ys=y, zs=z)
    ax.plot(xs=x1, ys=y1, zs=z1)
    ax.plot(xs=x2, ys=y2, zs=z2)
    ax.plot(xs=x3, ys=y3, zs=z3)
    plt.show()


if __name__ == '__main__':


    stripes = 201
    distance = 100.0/(stripes-1)
    print("distance:", distance)
    speed = 10
    #what is ok. speed? (5?)
    r = 1
    postions = np.linspace(0, 100, stripes)
    print(postions)
    # pttrn = []
    # for rep in range(stripes):
    #     for _ in range(2**rep):
    #         pttrn.append(postions[rep])
    # pttrn = np.array(pttrn)
    # print(pttrn)

    # arr = pattern2array3D(
    #     rate=r,
    #     max_dots=100,
    #     T = 0.01,
    #     pattern = postions,
    #     intensity = 2**np.array(range(1, stripes +1)),
    #     dot_size = [1.5, 1.5, 3],
    #     y_range = 100,
    #     z_range = 3, #TODO change it to the number of layers
    #     p0 = np.array([0, 0, 0]),
    #     dx = 1*r,
    #     dz = 1)
    speed_nr = 2
    intst = np.linspace(0.02, 1, len(postions))
    arr = pattern2array3D(
        #slowest was 0.25
        rate=r,
        max_dots=4000.0,
        T = 0.01,
        pattern = postions,
        intensity= np.ones_like(postions)*intst[speed_nr],
        # intensity=np.ones_like(postions),
        dot_size = [1, 5, 5],
        y_range = 100,
        z_range =100, #TODO change it to the number of layers
        p0 = np.array([0, 0, 0]),
        dx = speed*r,
        dz = speed*r)

    arr = np.array(arr)

    print(len(arr))
    # print(arr)
    check_array(arr, rate=r, max_speed=speed, max_points=3333*100) # speed have to be smaller that 10
    # array2file(r, arr, "array" + str(stripes) + "x" + str(1) + "lines-big.csv") # we have to print .csv
    array2file(r, arr, "3Dcube_"+str(speed_nr)+"_rate-"+str(r)+".lipp")
    print("file created")
    #TODO change file extension it in the documentation to lipp
    # array2file(5, arr, "cross.csv")

    # approximation_example()

    # speed_limit_example()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    N = 100
    x0 = arr[:,0]
    y0 = arr[:,1]
    z0 = arr[:,2]
    ax.plot(xs=x0, ys=y0, zs=z0, linewidth=1, alpha=0.5)
    ax.scatter(x0, y0, z0, marker="o", alpha=0.1)
    plt.show()