#!/usr/bin/env bash
# Script to convert files (and rearrange directories) form the retardace microscope.
# You probably do not need to read this.

for D in */ ; do
echo $D
cd "$D"
cp Pos0/* .
rm -R Pos0
tif2png.py
mkdir "tifs_$D"
mv *.tif "tifs_$D"
rm *.txt
cd ..
done