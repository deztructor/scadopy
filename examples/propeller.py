#!/usr/bin/env python3
from scad import *
from collections import namedtuple

size = Vector(100, 20)
ratio = size.y / size.x

wing = Vector(50, 50 * ratio, 100)
resolution = 0.1

def get_blade():
    return Import(file='blade-side.dxf') * resize(wing.xy)

get_outer = lambda: get_blade()
get_inner = lambda: get_outer() * offset(r=-1)

#wing_2d = outer() - inner()
wing_2d = get_outer()
#cap_3d = outer() * linear_extrude(2)
wing_3d = wing_2d * linear_extrude(wing.z)

side_wall = 2
get_cap_hole_3d = lambda: Cube(wing - Vector(16, 0, 0)) * translate([5, 0, 0])
get_base_3d = lambda: get_outer() * linear_extrude(side_wall)

def get_cap_3d_skeleton():
    res = get_inner() * linear_extrude(wing.z)
    return res - get_cap_hole_3d()

cap_3d = get_cap_3d_skeleton() * translate([0, 0, side_wall])
cap_3d += get_base_3d()
cap_3d -= Cube(wing) * translate([0, 0, 10])

holder_wall = 2
holder_hole = 1.5
cap_holder = Vector(holder_wall * 2 + holder_hole, holder_wall)
cap_holder = cap_holder.updated(z=cap_holder.x + side_wall)
def get_cap_holder_3d():
    return Cube(cap_holder) \
        - Cube(cap_holder) \
        * resize(cap_holder - [holder_wall * 2, 0, holder_wall * 2]) \
        * translate([holder_wall, 0, holder_wall])

dholder = (wing - cap_holder).updated(z=0)
cap_holder_3d = get_cap_holder_3d() * translate(dholder / [1.5, 2, 1])
cap_holder_3d += get_cap_holder_3d() * translate(dholder / [3, 2, 1])

cap_3d += cap_holder_3d
cap_3d *= mirror([1, 0, 0])

wing_3d = wing_3d * translate([0, 0, side_wall]) + get_base_3d()

dump(wing_3d, 'wing.scad')
dump(cap_3d, 'wing-support.scad')
