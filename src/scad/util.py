import collections
from functools import lru_cache
import math
from .scad import Circle, Square, translate
from . import geom


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

        cut = Square(size=[h, h])
        facet = Circle(r=r, **self.special) * translate(x=cir_pos, y=cir_pos)
        return cut - facet
