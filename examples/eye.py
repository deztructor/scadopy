#!/usr/bin/env python3

import math
from scad import *


inner = geom.Round(d=24)
wall = 2
ant_cam_depth = 3.7
pupil = geom.Round(d=7)

def _cylinder(*args, **kwargs):
    return Cylinder(*args, fs=0.05, **kwargs)

def eye_cavity():
    return Sphere(d=inner.d)

def inner_half_cube():
    return Cube([inner.d, inner.d, inner.r]) * translate([-inner.r, -inner.r])

def holes():
    r = 1
    hole = lambda x, y: Cube([r * 2, r * 2, inner.d]) * translate(x=x - r, y=y - r)
    off = inner.r - r - 1
    return (
        hole(-off, -off) + hole(-off, off) + hole(off, -off) + hole(off, off)
    ) * translate(z=-0.5)

def basic_eye_bottom():
    return (
        inner_half_cube() * translate(z=-inner.r)
        - eye_cavity()
    ) * translate(z=inner.r)

def eye_bottom():
    floor_h = 1
    base = Cube([inner.d, inner.d, floor_h]) * translate([-inner.r, -inner.r, 0])
    return (
        basic_eye_bottom() * translate(z=floor_h)
        + base
        - holes()
        + get_frame(Vector(inner.d, inner.d), 0, 0.5, inner.r + floor_h)
    )

def get_frame(xy, offset, wall, height):
    return (
        Cube((xy + offset * 2 + wall * 2).updated(z=height), center=True)
        - Cube((xy + offset * 2).updated(z=height), center=True)
    ) * translate(z=height / 2)

def iris():
    d = inner.d + 1
    max_h = 7
    size = Vector(inner.d, inner.d)
    offset = Vector(1, 1)

    return (
        Cube(size.updated(z=1) + offset * 2, center=True) * translate(z=0.5)
        - _cylinder(r=pupil.r, h=1)
        + get_frame(size, offset, 1, max_h)
    ) - holes()

def eye_top():
    body = (
        basic_eye_bottom()
        - inner_half_cube() * translate(z=ant_cam_depth - inner.r)
    ) * translate(z=-ant_cam_depth)
    return (
        body
        + get_frame(Vector(inner.d, inner.d), 0, 0.5, inner.r - ant_cam_depth)
        + get_frame(Vector(inner.d, inner.d), 0.5, 1, inner.r)
        - holes()
    )


with Scene() as scene:
    scene \
        << eye_bottom() \
        << eye_top() * translate(x=inner.d + 4) \
        << iris() * translate(x=-(inner.d + 4))
