"""
Created on 31.05.17

@author: miska
"""
import math
import numpy as np


def create_array1D(rate, max_dots, T, p0, pattern, intensity, reverse=False):
    """"Create a list of positions at which the plate should be at the uniform
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

    print("number of entries read between pulses:", rate/T)
    max_time = T*max_dots
    max_repeat = max_time/rate
    array = []
    if reverse:
        pattern = pattern[::-1]
        intensity = intensity[::-1]
    for x, i in zip(pattern, intensity):
        p = list(p0)
        p[0] += x
        for _ in range(math.floor(i*max_repeat)):
            array.append(p)
    return array


def step_pyramid_grating(steps_size, height, period, length):
    """"Calculate blob positions for pyramidal grating

        step_size - distance between neighbouring blobs
        height - number of blobs per pyramid
        period - distance between beginnings of pyramids
        length - leght of the plate

        Returns beginnings of blobs """""

    base = np.arange(0, length, period)
    factor = len(base)
    base = np.repeat(base, height, axis=0)
    positions = np.array(list(range(height)) * factor) * steps_size
    return positions + base
