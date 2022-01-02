# Rubik's Cube python

## Command line interface
To start execute the file CubeTextInterface.py

The avilable movements are:
r (right), l (left), u (up), d (down), f (front), b (back), x (rotate around the x axis),y (rotate around the y axis), z (rotate around the z axis).

By default any given turn will be clockwise.
For a given movement [a] if you write [a'] it will be inverted, if you write [a2] the movement will be repeated 2 times.

You can do a sequence of movements at once by writing them in succecion like this: r'l2ul'ru'f2

## Computer vision system
To start execute the file solve_IRL_cube.py

You will need a webcam or at least a camera connected to your computer

It's divided into 3 steps
  - Calibration: point the camera to the 6 faces of the cube to detect the color layout, follow the arrows to detect all the faces in order
  - Solving: an algorithm will try to solve your cube, this might take some time so don't be too impatient
  - Showing solution: you will be shown the steps required to solve your cube, follow them to get your cube solved
