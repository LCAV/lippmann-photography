
from printing_tools import *


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

    def block_position(self, t, space_resolution, space_range, i3d):
        """"This is helper function for self.approximate
        which returns position of i3D neighbour of the curve at time t
        (where i3D is a 3D index)"""""
        nbh = (np.array(range(space_range)) - np.floor(space_range / 2)) \
            * space_resolution
        p = np.floor(self.position(t) / space_resolution) \
            * space_resolution
        return p + np.array([nbh[i3d[0]], nbh[i3d[1]], nbh[i3d[2]]])

    @staticmethod
    def two_segments_distance(p0, p1, s1, s2):
        a = p0 - s1
        b = p1 - s2
        return (a.dot(a) + a.dot(b) + b.dot(b)) / 3.0

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
        i3d = []
        for xi in range(space_range):
            for yi in range(space_range):
                for zi in range(space_range):
                    i3d.append((xi, yi, zi))
                    paths[xi, yi, zi] = list([positions[0]])

        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            new_paths = paths
            new_errors = errors
            for i in i3d:
                p1 = self.block_position(t1, space_resolution, space_range, i)
                min_error = math.inf
                best_j = (0, 0, 0)
                for j in i3d:
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

        for i in i3d:
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


def approximation_example():
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(0, 1, 100)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    some_line = NicelyParameterized(x, y, z)

    error, path = some_line.approximate(0.1, 10, space_range=3)
    print(error)
    print(path.shape)
    to_plot = np.asarray(some_line.plot())
    #
    plot_pattern([path, to_plot], single=False)


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
