#!/usr/bin/env python3

import math
from scad import *


def _cylinder(*args, **kwargs):
    return Cylinder(*args, fs=0.05, **kwargs)

def blade():
    return (
        _cylinder(h=30, r=20)
        - _cylinder(h=30, r=19)
        - Cube([20, 20, 30])
        - Cube([20, 20, 30]) * rotate(az=90)
        - Cube([20, 20, 30]) * rotate(az=90)
        - _cylinder(r=60, h=50, center=True) * rotate(ax=90) * translate(z=60, x=-30)
    ) * translate(x=-20) * rotate(az=-90)

def rotated_blade(a):
    return blade() * rotate(az=a)

blades = (
    blade() * translate(x=-2)
    + rotated_blade(a=90) * translate(y=-2)
    + rotated_blade(a=180) * translate(x=2)
    + rotated_blade(a=270) * translate(y=2)
)

with Scene() as scene:
    scene << blades + (_cylinder(r=2.5, h=30) - _cylinder(r=1.5, h=30))

