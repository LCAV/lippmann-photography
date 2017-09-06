import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
# this is used for plots 3D
from mpl_toolkits.mplot3d import Axes3D
from printing_tools import *

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


def color_pattern(plot_stuff, wlen, speed, axis, layers=100, harm=0, n0=1.45, z_limit=100):
    if axis == "z":
        dz = (2 * harm + 1) * wlen / (2 * n0)
        z_steps = int(math.floor(z_limit / dz))
        positions = np.linspace(0, 100, layers)

        return positions, z_steps, dz
    if axis == "x":
        dx = (2 * harm + 1) * wlen / (2 * n0)
        positions = np.arange(0, 100, dx)
        dz = z_limit/(layers-1)

        return positions, layers, dz

    print("you can choose only x and z axis :/")

def heightTestPattern(plot_stuff, speed, line_lenght, steps):
    r = 1
    ll = line_lenght

    arr = heightTest(
        p0=np.array([0, 0, 0]),
        rate=r,
        line_len=ll,
        z_steps=steps,
        speed_limit=speed)

    plot_arr = arr

    if plot_stuff:
        plot_arr = np.array(plot_arr)
    arr = np.array(arr)

    print("full", len(arr))
    if plot_stuff:
        print("without markers", len(plot_arr))
    # print(arr)
    check_array(plot_arr, rate=r, max_speed=speed + 0.01,
                max_points=3333 * 100, max_speed_change=speed)
    array2file(r, arr, "heightTest_length" + str(ll) + "_steps_" + str(steps) + "_speed.lipp")
    print("file created")

    if plot_stuff:
        plot_pattern(plot_arr)


if __name__ == '__main__':
    plot_stuff = False
    speed = 5
    rate = 1
    wlen = 0.521
    layers = 20
    axis = "z"
    positions, z_steps, delta_z = color_pattern(plot_stuff, wlen, speed, axis=axis, layers=layers)

    plot_arr, arr = pattern2array3DPair(
        p0=np.array([0, 0, 100]),
        rate=rate,
        pattern=positions,
        speed=np.ones_like(positions) * speed,
        z_steps=z_steps,
        delta_z=delta_z)

    if plot_stuff:
        plot_arr = np.array(plot_arr)
    arr = np.array(arr)

    print("full length:", len(arr))

    if plot_stuff:
        print("without markers", len(plot_arr))

    check_array(plot_arr, rate=rate, max_speed=speed,
                max_points=3333 * 1000, max_speed_change=speed)
    array2file(rate, arr, "color_"+ axis +"_" + str(wlen) + "_wlen_" + str(speed) + "_speed.lipp")

    print("file created")

    if plot_stuff:
        plot_pattern(plot_arr)



