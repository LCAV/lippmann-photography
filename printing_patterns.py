import math
from printing_tools import *


def color_pattern(wavelength, axis_name, layers=100, harm=0, n0=1.45, z_limit=100):
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

    arr = height_test(
        rate=rate,
        line_len=line_length,
        z_steps=layers_,
        speed_limit=speed)

    plot_arr = arr

    # positions, z_steps, delta_z = color_pattern(plot_stuff, wlen, speed, axis=axis, layers=layers)

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
