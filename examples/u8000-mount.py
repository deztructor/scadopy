#!/usr/bin/env python3
from scad import *
from collections import namedtuple

HoleData = namedtuple('HoleData', 'd h offset')
Tooth = namedtuple('Tooth', 'x h')
width = 36
height = 8
length = 37
slope = 4
rubber_height = 2
roof_thick = 3
hang = 2
bevel = 12
tooth = Tooth(3, 3)
shoe_thick = Vector(2, 2, roof_thick)

top_hole = HoleData(10, 1, 0)
# should be wider
through_hole = HoleData(6.5, height + 1, 0)

#shoe_f = Cube(size=shoe)
shoe_points = [
    [0, 0], [length, 0],
    [length - slope, height], [slope, height]
]
shoe_hole_points = [
    [shoe_thick.x, 0], [length - shoe_thick.x, 0],
    [length - shoe_thick.x - slope, height - shoe_thick.z],
    [slope + shoe_thick.x, height - shoe_thick.z]
]
hole_width = width - shoe_thick.y * 2

shoe_2d = Polygon(shoe_points)
shoe_hole_2d = Polygon(shoe_hole_points)
shoe_3d = shoe_2d * linear_extrude(height=width)
shoe_hole_3d = shoe_hole_2d * linear_extrude(hole_width)
shoe_3d = shoe_3d - shoe_hole_3d * translate([0, 0, shoe_thick.y])
shoe_3d = shoe_3d * translate([0, -height, 0]) * rotate([-90, 0, 0]) \
          * translate([-length / 2, -width / 2, 0])

base = Vector(length - slope, width + hang * 2, rubber_height)
base_hole = base - Vector(bevel, bevel, 0)
base_3d = (Cube(size=base, center=True) \
          - Cube(size=base_hole, center=True)) \
          * translate([0, 0, rubber_height / 2])

base_hole_offset = base_hole / Vector(2, 2, 1)
base_hole_corners = [
    base_hole_offset,
    base_hole_offset * [-1, 1, 1],
    base_hole_offset * [1, -1, 1],
    base_hole_offset * [-1, -1, 1],
]

def make_hole(data):
    res = Cylinder(h=data.h, d=data.d, fn=20)
    return res * translate([0, 0, data.offset]) if data.offset != 0 else res

shoe_3d = shoe_3d - make_hole(through_hole)

base_offset = base.updated(y=0, z=0)

def leg(r, leg_h, location):
    return Cylinder(r=r, h=leg_h, center=True, fn=20) \
        * translate(location)

shoe_3d = shoe_3d - Union(
    *(leg(1.5, height + 1, offset) for offset in base_hole_corners)
)

base_3d = base_3d + Union(
    *(leg(1, 4, offset) for offset in base_hole_corners)
)

HolderPivot = namedtuple('HolderPivot', 'hole bottom top')
Holder = namedtuple('Holder', 'pivot handle_len')
holder = Holder(HolderPivot(2.5, 7.5 + slope, 7.5), 25)

holder_points = [
    [0, 0], [holder.pivot.bottom, 0],
    [holder.pivot.top, height], [0, height]
]

get_holder_2d = lambda: Polygon(holder_points)
holder_3d = get_holder_2d() * rotate_extrude()

def get_holder_hole():
    return Cylinder(r=holder.pivot.hole, h=height, fn=20)


holder_3d -= get_holder_hole()
holder_cut_3d = Cube([holder.pivot.bottom, holder.pivot.bottom * 3, height]) \
                * translate([holder.pivot.hole + 2.5, -holder.pivot.bottom, 0]) \
                * rotate([0, 0, -75])
holder_3d -= holder_cut_3d

holder_nut_3d = Cylinder(d=9.5, h=5, fn=6) * rotate([0, 0, 15])

handle_3d = get_holder_2d() * linear_extrude(holder.handle_len) \
            * rotate([90, 0, 180])
handle_cut_3d = Cube([holder.pivot.hole, holder.handle_len, height]) \
                * translate([-holder.pivot.hole, 0, 0])
handle_3d -= handle_cut_3d
#get_holder_hole()
holder_3d += handle_3d

holder_3d -= holder_nut_3d

holder_3d = holder_3d * rotate([0, 180, 0]) * translate([0, 0, height])

dump(shoe_3d * translate(-base_offset), 'u8000-shoe.scad')
dump(base_3d * translate(-base_offset), 'u8000-base.scad')
dump(holder_3d, 'u8000-holder.scad')
