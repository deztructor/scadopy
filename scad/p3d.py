import enum
import collections

from cor.adt.record import(
    ExtensibleRecord,
    Record,
)
from cor.adt.operation import (
    convert,
    expect_type,
    provide_missing,
    skip_missing,
)
from cor.adt.hook import (
    field_invariant,
    field_aggregate,
)
from scad import geom

class Material(enum.Enum):
    PLA = 'pla'
    ABS = 'abs'
    PETG = 'petg'
    PC = 'polycarbonate'
    Nylon = 'nylon'


class Wall(Record):
    perim = convert(float)
    w = convert(float)


def _check_layer(printer, _, layer):
    if layer > printer.nozzle * 0.75:
        raise Exception("Nozzle is too small for the layer height {}".format(layer))


def _provide_missing_wall(printer, field_name, value):
    if value:
        return None
    nozzle = printer.nozzle
    return field_name, Wall(perim=nozzle * 1.125, w=nozzle)


class Printer(Record):
    layer = convert(float) << field_invariant(_check_layer)
    material = provide_missing(Material.PLA) >> expect_type(Material)
    nozzle = provide_missing(0.4) >> convert(float)
    wall = skip_missing >> expect_type(Wall) << field_aggregate(_provide_missing_wall)
    plate = provide_missing(geom.Vector(x=250, y=210)) >> convert(geom.Vector)

    def hole_size(self, size, is_xy):
        return (
            0.9927 * size + 0.3602
            if is_xy
            else 1.0155 * size + 0.2795
        )
