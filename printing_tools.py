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


def check_array(array, rate, max_speed=10, max_speed_change=100,
                max_points=3333, min_pos=0, max_pos=100):
    """""Checks if array is OK to print.

    Checks the number of points, the speed limit, the acceleration limit
    and if the coordinates are in the right cube

    array       - array of length-3 arrays (in microns)
    rate        - rate at which to move between points (in milliseconds)
    max_speed   - max. speed allowed by piezo (in microns over milliseconds)
    max_acc     - max. acceleration allowed by piezo (in microns over ms^2)
    max_points  - max. number of points which can be printed in one go
    min_pos     - min. position possible for the piezo
    max_pos     - max. position possible for the piezo
    """""

    np.set_printoptions(precision=2, suppress=True)
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

        if any(np.abs(s2-s1)>max_speed_change):
            print("[Failed] Change in the speed", s2-s1,
                  "bigger than", max_speed_change,
                  "for points", p1, p2, "and", p3)
            print("(rate", rate, ")")
    if checks_OK:
        print("[Success] All checks passed :)")

if __name__ == '__main__':

    new_pattern = circle_array(radius=50, n=100, p0=[50, 0, 0])
    print(len(new_pattern))
    array2file(0.2, new_pattern, "circle_small.lipp")
