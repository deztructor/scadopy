import collections
from functools import partial
import math

from cor import is_around, Attrs


RightTriangleFields = collections.namedtuple('RightTriangleFields', 'a b c alpha beta')


class RightTriangle(RightTriangleFields):

    angles = {'alpha', 'beta'}
    sides = {'a', 'b', 'c'}
    params = set(RightTriangleFields._fields)

    def __new__(cls, **kwargs):
        fields = cls._resolve(kwargs)
        return super().__new__(cls, **fields._asdict())

    @classmethod
    def _resolve(cls, fields):
        ok = lambda x: x is not None and x
        d90 = math.pi / 2

        def raise_not_complete():
            raise ValueError('Incomplete set: {}'.format(fields))

        def _angles_from_sides(t, cant_be_done):
            if ok(t.a):
                if ok(t.b):
                    t.alpha = math.atan(t.a / t.b)
                elif ok(t.c):
                    t.alpha = math.asin(t.a / t.c)
                else:
                    return cant_be_done()
            elif ok(t.b) and ok(t.c):
                t.alpha = math.acos(t.b / t.c)
            else:
                return cant_be_done()
            t.beta = d90 - t.alpha
            return t

        def _calc_sides(t, cant_be_done):
            assert ok(t.alpha) and ok(t.beta)
            if ok(t.c):
                if not ok(t.a):
                    t.a = math.sin(t.alpha) * t.c
                if not ok(t.b):
                    t.b = math.cos(t.alpha) * t.c
                else:
                    assert is_around(math.hypot(t.a, t.b), t.c)
            elif ok(t.a):
                t.c = t.a / math.sin(t.alpha)
                if not ok(t.b):
                    t.b = t.a / math.tan(t.alpha)
            elif ok(t.b):
                t.c = t.b / math.cos(t.alpha)
                t.a = math.tan(t.alpha) * t.b


        def _calc_angles(t, cant_be_done):
            if ok(t.alpha):
                if not ok(t.beta):
                    t.beta = d90 - t.alpha
                else:
                    assert is_around(t.alpha + t.beta, d90)
            elif ok(t.beta):
                t.alpha = d90 - t.beta
                assert is_around(t.alpha + t.beta, d90)
            else:
                _angles_from_sides(t, raise_not_complete)

        t = Attrs.from_map(cls.params, fields)

        _calc_angles(t, raise_not_complete)
        _calc_sides(t, raise_not_complete)

        return RightTriangleFields(*t.as_args(RightTriangleFields._fields))

    @classmethod
    def is_enough_params(cls, names):
        names = set(names)
        sides = len(cls.sides & names)
        angles = len(cls.angles & names)
        return sides >= 2 or (angles > 0 and sides > 0)


class Round(collections.namedtuple('Round', 'r d')):
    def __new__(cls, r=None, d=None):
        if r is None:
            if d is None:
                raise Exception('r or d?')
            r = d / 2
        elif d is None:
            d = r * 2
        else:
            raise Exception('r _or_ d')

        return super().__new__(cls, r, d)

    def __add__(self, v):
        if not isinstance(v, Round):
            raise Exception('Needs Round')
        return Round(d=self.d + v.d)

    def __sub__(self, v):
        if not isinstance(v, Round):
            raise Exception('Needs Round')
        return Round(d=self.d - v.d)

    def __truediv__(self, v):
        return Round(d=self.d / 2)

    def __mul__(self, v):
        return Round(d=self.d * 2)


class Sector:
    params = {'alpha', 'beta', 'r', 'chord', 'dcenter', 'dedge'}
    angles = ('alpha', 'beta')
    lines = ('r', 'chord', 'dcenter', 'dedge')

    _params_table_rtri = {
        'chord': lambda x: ('a', x / 2),
        'dcenter': lambda x: ('b', x),
        'alpha': lambda x: ('alpha', x / 2),
        'beta': lambda x: ('beta', x),
        'r': lambda x: ('c', x),
        'dedge': lambda x: ('dedge', x),
    }

    def __init__(self, **kwargs):
        ok = lambda x: x is not None and x
        def _params_right_tri():
            params = dict(self._params_table_rtri[k](v) for k, v in kwargs.items())
            return Attrs.from_map(self.params | RightTriangle.params, params)

        keys = set(kwargs.keys())
        if not keys.issubset(self.params):
            raise ValueError('Expecting any of {}, got {}'.format(self.params, kwargs))

        p = _params_right_tri()
        set_names = [k for k, v in p.as_dict().items() if v is not None]
        need_more_params = not RightTriangle.is_enough_params(set_names)
        if ok(p.dedge):
            if need_more_params:
                if ok(p.a):
                    p.c = (p.dedge + p.a * p.a / p.dedge) / 2
                elif ok(p.c):
                    p.b = p.c - p.dedge
                elif ok(p.b):
                    p.c = p.b + p.dedge
        elif need_more_params:
            raise ValueError('Need more params, got {}'.format(p))

        self._t = RightTriangle(**p.as_dict())

    @property
    def alpha(self):
        return self._t.alpha * 2

    @property
    def chord(self):
        return self._t.a * 2

    @property
    def r(self):
        return self._t.c

    @property
    def dedge(self):
        return self._t.r - self._t.b

    @property
    def dcenter(self):
        return self._t.b
