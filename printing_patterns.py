import math
from printing_tools import *


def circle_array(radius, n, p0):
    """"Simple circle, or rather regular polygon with n vertices"""""
    array = []
    for phi in np.linspace(0, 2 * np.pi, n):
        p = list(p0)
        p[0] += np.sin(phi) * radius
        p[1] += (1 + np.cos(phi)) * radius
        array.append(list(p))
    return array


def color_array(wavelength, axis_name, layers=100, harm=0, n0=1.45, z_limit=100):
    if axis_name == "z":
        dz = (2 * harm + 1) * wavelength / (2 * n0)
        z_steps = int(math.floor(z_limit / dz))
        positions = np.linspace(0, 100, layers)

        return positions, z_steps, dz
    if axis_name == "x":
        dx = (2 * harm + 1) * wavelength / (2 * n0)
        positions = np.arange(0, 100, dx)
        dz = z_limit/(layers-1)

        return positions, layers, dz

    print("you can choose only x and z axis :/")


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


def height_test_array(rate, line_len, z_steps,
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


if __name__ == '__main__':

    # general parameters
    plot_stuff = True
    speed = 2
    rate = 1

    # color pattern parameters
    wlen = 0.521
    layers_ = 20
    axis = "x"

    # height pattern parameters
    line_length = 20

    arr = height_test_array(
        rate=rate,
        line_len=line_length,
        z_steps=layers_,
        speed_limit=speed)

    plot_arr = arr

    # positions, z_steps, delta_z = color_array(plot_stuff, wlen, speed, axis=axis, layers=layers)

    # plot_arr, arr = pattern2array3d_pair(
    #     p0=np.array([0, 0, 100]),
    #     rate=rate,
    #     pattern=positions,
    #     speed=np.ones_like(positions) * speed,
    #     z_steps=z_steps,
    #     delta_z=delta_z)

    if plot_stuff:
        plot_arr = np.array(plot_arr)
    arr = np.array(arr)

    print("full length:", len(arr))

    if plot_stuff:
        print("without markers", len(plot_arr))

    check_array(plot_arr, rate=rate, max_speed=speed+0.01,
                max_points=3333 * 1000, max_speed_change=speed+0.01)
    # array2file(rate, arr, "color_"+ axis +"_" + str(wlen) + "_wlen_" + str(speed) + "_speed.lipp")

    print("file created")

    if plot_stuff:
        plot_pattern(plot_arr)
