#!/usr/bin/env python3
import itertools
from collections import namedtuple
from scad import *

import sys

Row = namedtuple('Row', 'begin gen_offsets')
Cell = namedtuple('Cell', 'width beam')

def calc_row(cell, width):
    if cell.beam > cell.width:
        raise ValueError('Beam width should be < cell size')
    count = int(width / cell.width)
    if count == 0:
        raise Exception('No cells fit into grid')

    row_w = cell.width * count
    begin = (width - row_w) / 2
    gen_offsets = lambda: itertools.accumulate(
        itertools.chain(
            (begin,),
            itertools.repeat(cell.width, count)
        ))
    return Row(begin, gen_offsets)

def get_grid(cell, xbeam, ybeam):
    row_y = calc_row(cell, xbeam.x)
    row_x = calc_row(cell, ybeam.y)
    items = itertools.chain(
        (Cube(size=ybeam) * translate([pos, 0, 0])
         for pos in row_y.gen_offsets()),
        (Cube(size=xbeam) * translate([0, pos, 0])
         for pos in row_x.gen_offsets())
    )
    return Union(*items)

def get_scene_items():
    lx, lyl, lys = 27.0, 54.0, 45.0
    cell = Cell(5.0, 1.0)
    h = 3.0
    beams_y = (Vector(cell.beam, lyl, h), Vector(cell.beam, lys, h))
    xbeam = Vector(lx, cell.beam, h)

    grids = [get_grid(cell, xbeam, beam_y) for beam_y in beams_y]
    return [
        grids[1],
        grids[0] * translate([xbeam.x + 3, 0, 0]),
        grids[1] * translate([-xbeam.x - 3, 0, 0])
    ]


with Scene() as scene:
    scene << get_scene_items()
