#!/Users/miska/anaconda/bin/python

import argparse
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, splitext


def silly_tif2png(name=None, directory=None, cmap=None):
    """ Convert tif to png using matplotlib.
      name      - name of the file to convert,
            if None then coverts all files in the directory
      directory - name of the directory containing files to convert,
            if None then uses current directory
      cmap      - matplotlib colormap used to convert image,
            if None grayscale is used"""

    if directory is None:
        directory = "."
    if cmap is None:
        cmap = "gray"

    if name is None:
        files_names = [f for f in listdir(directory) if isfile(f)]
        for f in files_names:
            f_name, ext = splitext(f)
            if ext == ".tif":
                img = plt.imread(join(directory, f))
                plt.imsave(fname=join(directory, f_name+".png"),
                           arr=img, cmap=cmap, format="png")
    else:
        f_name, ext = splitext(name)
        if ext == ".tif":
            img = plt.imread(join(directory, name))
            plt.imsave(fname=join(directory, f_name + ".png"),
                       arr=img, cmap=cmap, format="png")
        else:
            print("The file is not .tif")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", help="name of the file to convert")
parser.add_argument("-d", "--input_directory",
                    help="name of the directory with files")
parser.add_argument("-c", "--color_map", help="matplotlib colormap of choice")
args = parser.parse_args()

silly_tif2png(name=args.input_file,
              directory=args.input_directory,
              cmap=args.color_map)
