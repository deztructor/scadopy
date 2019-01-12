import collections
import functools
import math
from types import SimpleNamespace as _NS

from . import scad
from . import geom
from . import scad

class RoundFacet(collections.namedtuple('RoundFacet', 'h r special')):
    def __new__(cls, h, r=None, **kwargs):
        if r is None:
            r = h
        elif r < h:
            raise ValueError('Need r >= h, got r={}, h={}'.format(r, h))

        if not ({'fs', 'fa', 'fn'} & set(kwargs.keys())):
            kwargs =  {'fs': (h * h / 10 / r)}

        return super().__new__(cls, h, r, kwargs)

    # TODO cache
    def get(self):
        h, r = self.h, self.r
        hyp = math.hypot(h, h)
        s = geom.Sector(r=r, chord=hyp)
        cir_pos = h / 2 - s.dcenter * math.sin(math.radians(45))

        cut = scad.Square(size=[h, h])
        facet = scad.Circle(r=r, **self.special) * scad.translate(x=cir_pos, y=cir_pos)
        return cut - facet


class _D:
    DEBUG = False

    def __init__(self, clr):
        self._color = clr or 'red'

    def __rmul__(self, v):
        return (
            v * scad.color(self._color, a=0.5) if self.DEBUG
            else scad.placeholder((v * scad.color(self._color, a=0.5)).dimensions)
        )

def debug(clr=None):
    return _D(clr)


def set_debug(v):
    _D.DEBUG = v


def _rect_slice(alpha, delta, get_rect):
    return (
        get_rect() - (get_rect() * scad.scale(x=2) * scad.rotate(az=delta))
        if delta < 90
        else get_rect()
    ) * scad.rotate(az=alpha)


def sector_cut(alpha, r):
    if alpha > 360:
        raise scad.Error('alpha <= 360, got {}'.format(alpha))
    return scad.Union(*(
            _rect_slice(*slices, lambda: scad.Square([r, r]))
            for slices in geom.gen_slices(alpha)
    ))


def cylinder_cut(alpha, r, h):
    if alpha > 360:
        raise Error('alpha <= 360, got {}'.format(alpha))
    return scad.Union(*(
            _rect_slice(*slices, lambda: scad.Cube([r, r, h]))
            for slices in geom.gen_slices(alpha)
    )) * scad.rotate((360 - alpha) / 2)


class CountersinkHole(collections.namedtuple('CountersinkBolt', 'r_out h_head r_in h' )):
    def get_3d(self, **kwargs):
        x = _NS(a=0, b=self.r_in, c=self.r_out)
        y = _NS(a=0, b=-self.h_head, c=-self.h)
        points = ([x.a, y.a], [x.c, y.a], [x.b, y.b], [x.b, y.c], [x.a, y.c])
        return scad.Polygon(points) * scad.rotate_extrude(**kwargs)
