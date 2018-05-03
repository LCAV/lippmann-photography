#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 12:44:29 2018

@author: gbaechle
"""

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.pyplot import figure, show


def quad(plane='xy', origin=None, width=1, height=1, depth=0):
    u, v = (0, 0) if origin is None else origin

    plane = plane.lower()
    if plane == 'xy':
        vertices = ((u, v, depth),
                    (u + width, v, depth),
                    (u + width, v + height, depth),
                    (u, v + height, depth))
    elif plane == 'xz':
        vertices = ((u, depth, v),
                    (u + width, depth, v),
                    (u + width, depth, v + height),
                    (u, depth, v + height))
    elif plane == 'yz':
        vertices = ((depth, u, v),
                    (depth, u + width, v),
                    (depth, u + width, v + height),
                    (depth, u, v + height))
    else:
        raise ValueError('"{0}" is not a supported plane!'.format(plane))

    return np.array(vertices)


def grid(plane='xy',
         origin=None,
         width=1,
         height=1,
         depth=0,
         width_segments=1,
         height_segments=1):
    u, v = (0, 0) if origin is None else origin

    w_x, h_y = width / width_segments, height / height_segments

    quads = []
    for i in range(width_segments):
        for j in range(height_segments):
            quads.append(
                quad(plane, (i * w_x + u, j * h_y + v), w_x, h_y, depth))

    return np.array(quads)


def cube(plane=None,
         origin=None,
         width=1,
         height=1,
         depth=1,
         width_segments=1,
         height_segments=1,
         depth_segments=1):
    plane = (('+x', '-x', '+y', '-y', '+z', '-z')
             if plane is None else
             [p.lower() for p in plane])
    u, v, w = (0, 0, 0) if origin is None else origin

    w_s, h_s, d_s = width_segments, height_segments, depth_segments

    grids = []
    if '-z' in plane:
        grids.extend(grid('xy', (u, w), width, depth, v, w_s, d_s))
    if '+z' in plane:
        grids.extend(grid('xy', (u, w), width, depth, v + height, w_s, d_s))

    if '-y' in plane:
        grids.extend(grid('xz', (u, v), width, height, w, w_s, h_s))
    if '+y' in plane:
        grids.extend(grid('xz', (u, v), width, height, w + depth, w_s, h_s))

    if '-x' in plane:
        grids.extend(grid('yz', (w, v), depth, height, u, d_s, h_s))
    if '+x' in plane:
        grids.extend(grid('yz', (w, v), depth, height, u + width, d_s, h_s))

    return np.array(grids)


if __name__ == '__main__':
    plt.close('all')
    
    canvas = figure(figsize=(3.45, 2.5))
    axes = Axes3D(canvas)
    
    quads = cube(width_segments=10, height_segments=10, depth_segments=10)
    
    # You can replace the following line by whatever suits you. Here, we compute
    # each quad colour by averaging its vertices positions.
    RGB = np.average(quads, axis=-2)
    # Setting +xz and -xz plane faces to black.
    RGB[RGB[..., 1] == 0] = 0
#    RGB[RGB[..., 1] == 1] = 0
    # Adding an alpha value to the colour array.
    RGBA = np.hstack((RGB, np.full((RGB.shape[0], 1), 1)))
    
    collection = Poly3DCollection(quads)
    collection.set_color(RGBA)
    axes.add_collection3d(collection)