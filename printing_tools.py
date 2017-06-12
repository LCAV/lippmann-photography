"""
Created on 31.05.17

@author: miska
"""
import math
import numpy as np

""" Naming conventions in this file (should we use classes or namespaces?)
    _pattern - returns positions of the dots
    _array   - returns positions 3D at equal timestamps"""


def pattern2array1D(rate, max_dots, T, p0, pattern, intensity, reverse=False):
    """" Create a list of positions at which the plate should be at the uniform
    time intervals (rate). This is approximation that may print off up to one
    dot per stack of dots (but his should not matter if number of dots printed
    in one place is of order of 50).

    rate        - rate at which the data is red: Range = 5ms - 1/30ms the
                    shortest time in which the plate can move
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
        print(delimiter.join((map(str, pattern[0]))), file=file)
        for line in pattern:
            print(delimiter.join(map(str, line)), file=file)


if __name__ == '__main__':

    new_pattern = circle_array(radius=50, n=100, p0=[50, 0, 0])
    print(len(new_pattern))
    array2file(0.2, new_pattern, "circle_small.lipp")
