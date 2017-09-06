"""
Created on 31.05.17

@author: miska
"""
import math
import numpy as np

""" Naming conventions in this file (should we use classes or namespaces?)
    _pattern - returns positions of the dots
    _array   - returns positions 3D at equal timestamps

    all 3D arrays should be in the format Nx3, because that's the format
    we print later"""


class Trajectory:
    """Abstract class which represent trajectory to be approximated using
    piecewise straight line, with knots at grid points"""

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
        self.pos = np.zeros((len(xs), 3))
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


def move_slow(p0, p_curr, s_max, ds, coord, margin):

    full = True
    if s_max - p_curr[coord] == 0:
        return p_curr, []

    if abs(s_max-p_curr[coord]) < 2*margin:
        full = False
        margin = (s_max-p_curr[coord])/2

    arr = []
    direction = 1
    if p_curr[coord] - s_max > 0:
        direction = -1
    s_0 = p_curr[coord]
    a = ds**2/(4*margin)
    t_max = 2*margin/ds
    p = list(p_curr)
    for t in np.arange(0, t_max, 1.0):
        s = a*t**2
        p[coord] = direction*s + s_0
        arr.append(list(p0+p))

    # acelerate to marigin,
    if full:
        new_beg = list(p_curr)
        new_beg[coord] += direction*margin
        p, tmp_arr = move(p0, new_beg, s_max - direction*margin, ds, coord)
        arr += tmp_arr
    # descacelerate

    for t in np.arange(t_max, 0, -1.0):
        s = a*t**2
        p[coord] = -direction*s + s_max
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


def pattern2array3D(rate, pattern, speed, z_steps, delta_z,
                    speed_limit=10, p0=np.array([0, 0, 0]),
                    y_range=100, z_range=100, point_limit = 3333,
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
            p, tmp_arr = move(p0, p, pattern[x], ds=np.min([ds, 0.3*dx]), coord=0)
            if cut and counter + len(tmp_arr) > point_limit:
                array.append([-1, -1, -1])
                counter = len(tmp_arr)
            else:
                counter += len(tmp_arr)
            array += list(tmp_arr)
            p[0] = pattern[x]
            p, tmp_arr = move_slow(p0, p, 0 if inverted else y_range, dy[x], coord=1, margin=margin)
            if cut and counter + len(tmp_arr) > point_limit:
                array.append([-1,-1,-1])
                counter = len(tmp_arr)
            else:
                counter += len(tmp_arr)
            array += list(tmp_arr)
        if z < z_steps - 1:
            # That's a hack! (0.5 ds)
            dz = np.min([ds, 0.3*delta_z])
            p, tmp_arr = move(p0, p, -(z+1)*delta_z, dz, coord=2)
            if cut and counter + len(tmp_arr) > point_limit:
                array.append([-1,-1,-1])
                counter = len(tmp_arr)
            else:
                counter += len(tmp_arr)
            array += list(tmp_arr)
            p[2] = -(z+1)*delta_z
            pattern = pattern[::-1]
    return array


def pattern2array3DPair(rate, pattern, speed, z_steps, delta_z,
                    speed_limit=10, p0=np.array([0, 0, 0]),
                    y_range=100, z_range=100, point_limit = 3333, margin=5):
    return (pattern2array3D(rate, pattern, speed, z_steps, delta_z,
                    speed_limit, p0,
                    y_range, z_range, point_limit, cut=False, margin=margin),
            pattern2array3D(rate, pattern, speed, z_steps, delta_z,
                    speed_limit, p0,
                    y_range, z_range, point_limit,
                    cut=True, margin=margin))


def heightTest(rate, line_len, z_steps,
                    speed_limit=10.0, p0=np.array([0.0, 0.0, 0.0]),
                    y_range=100.0, z_range=100.0, x_range=100.0, point_limit = 3333,
                    cut=False, margin=5.0):
    """This is a specific pattern to print in order to check if the height 
    is correct"""

    p = [0,0,0]
    array = [np.copy(p0)]
    ds = speed_limit * rate
    delta_z = z_range/z_steps
    counter = 0
    line_dist_x = (x_range - line_len) / z_steps

    for z in range(z_steps):
        y0 = p[1]
        x0 = p[0]
        p, tmp_arr = move_slow(p0, p, y0+line_len, ds, 1, margin)
        array += list(tmp_arr)
        p[1] = y0+line_len
        p, tmp_arr = move_slow(p0, p, y0+line_len*z/z_steps, ds, 1, margin)
        array += list(tmp_arr)
        p[1] = y0+line_len*z/z_steps
        p, tmp_arr = move_slow(p0, p, x0+line_len, ds, 0, margin)
        array += list(tmp_arr)
        p[0] = x0+line_len
        p, tmp_arr = move_slow(p0, p, x0, ds, 0, margin)
        array += list(tmp_arr)
        p[0] = x0
        p, tmp_arr = move_slow(p0, p, y0, ds, 1, margin)
        array += list(tmp_arr)
        p[1] = y0
        if z < z_steps - 1:
            p, tmp_arr = move(p0, p, x0 + line_dist_x, ds, 0)
            array += list(tmp_arr)
            p[0] = x0 + line_dist_x
            p, tmp_arr = move(p0, p, (z+1)*delta_z, ds, 2)
            array += list(tmp_arr)
            p[2] = (z+1)*delta_z
    return array


def intensity2speed(T, dot_size, intensity, max_dots):
    intensity /= np.max(intensity)
    return dot_size / (intensity * T * max_dots)


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
    checks_OK = True
    if len(array) > max_points:
        print("[Failed] To many points, expected:",
              max_points, "got:", len(array))
        checks_OK = False

    depth = 101

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
            checks_OK = False

        if p2[2] > depth + 0.01:
            print("[Failed] path is going down around point: ", p2)
            checks_OK = False
        depth = p2[2]

    if checks_OK:
        print("[Success] All checks passed :)")