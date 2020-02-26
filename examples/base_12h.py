#!/usr/bin/env python3

import sys
from scad import *
from collections import namedtuple
from functools import reduce, partial
from cor import compose

def ring(ri, ro, h, **kwargs):
    return Cylinder(r=ro, h=h, **kwargs) - Cylinder(r=ri, h=h, **kwargs)

HollowCube = namedtuple('HollowCube', 'inner outer combined')

Hole = namedtuple('Hole', 'size walls')

def get_hollow_cube(internal_size, walls):
    inner = Cube(internal_size)
    outer = inner.expanded(walls)
    return HollowCube(inner, outer, outer - inner * translate(walls))

def gen_pipes_ring(get_obj_and_size, angles, r):
    for angle in angles:
        obj, size = get_obj_and_size()
        yield obj * translate([-size[0] / 2, r, 0]) * rotate([0, 0, angle])

hole = Hole(size=[3, 6, 3], walls=[1, 1, 0])
radius = 30

get_hole_cube = partial(get_hollow_cube, hole.size, hole.walls)

select_combined = lambda x: (x.combined, x.outer.size)
select_outer = lambda x: (x.outer, x.outer.size)

get_pipe_and_size = compose(select_combined, get_hole_cube)
get_outer_and_size = compose(select_outer, get_hole_cube)

get_angles = lambda: range(0, 360, int(360 / 12))

gen_pipes = partial(gen_pipes_ring, get_pipe_and_size, get_angles(), radius)
gen_cubes = partial(gen_pipes_ring, get_outer_and_size, get_angles(), radius)

_, outer_rect_size = get_outer_and_size()

outer_r = radius + outer_rect_size[1]
rings = ((radius, radius + 2), (outer_r - 2, outer_r))
rings = [ring(*r, h=2, fa=0.2) for r in rings]

spokes = list(Cube([2, radius, 1]) * translate([-1, 0, 0]) * rotate(a)
              for a in get_angles())

class Cubes(Module):
    def get_object(self):
        return Union(*gen_cubes())

with Scene() as scene:
    solid_cubes_ring = Cubes()
    scene << solid_cubes_ring
    scene << gen_pipes() << Union(*spokes) - Cube(4, center=True)
    scene << Union(*rings) - solid_cubes_ring()

