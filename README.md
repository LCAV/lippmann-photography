# Lippmann Photography

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4256774.svg)](https://doi.org/10.5281/zenodo.4256774)

Code to analyse hyperspectral measurements historical Lippmann photographs and simulate the Lippmann proces. 

## Authors

Gilles Baechler, Michalina Pacholska, Adam Scholefield, Arnaud Latty from [LCAV](https://www.epfl.ch/labs/lcav/)

## Installation

Get this repository using:

    git clone  https://github.com/LCAV/lippmann-photography.git

You can install all standard python requirements it (at least) two ways:
 
1. using `pip` in you favourite Python 3 environment:
    ```
    pip install -r requirements.txt
    ```
2. using `conda`, that will create virtual environment for you:
    ```
    conda env create -f environment.yml
    ```

## Data
Data for this repository is available separately at 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4650243.svg)](https://doi.
org/10.5281/zenodo.4650243)

## Structure

To visualise results view `PNAS resutls.jpynb` with Jupyter Notebook. 
To generate data run `spectrum_recovery_pool.py`. Code for counting oscillations 
is exceptionally in MATLAB, not python in `oscillations_counting.m`.

## How to cite this code

Gilles Baechler, Michalina Pacholska, Adam Scholefield, & Arnaud Latty. (2020, 
November 7). LCAV/lippmann-photography (Version v0.1). Zenodo. 
http://doi.org/10.5281/zenodo.4256775

or see http://doi.org/10.5281/zenodo.4256775 for other formats.
    
## License

Copyright 2020 Gilles Baechler, Michalina Pacholska, Adam Scholefield, Arnaud Latty

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
