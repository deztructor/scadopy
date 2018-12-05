import collections
from collections import namedtuple, Iterable, Sequence
import functools
import math
import numbers

import cor


from cor import is_around, Attrs


class Variable:
    _counter = 0

    def __init__(self, value=None, name=None):
        type(self)._counter += 1
        self.name = name or type(self).__name__
        self.unique_name = '_{}#{}'.format(self.name, self._counter)
        self.value = value

    def __str__(self):
        return (self.name or '<unknown>') + '=' + (str(self.value) or '<?>')

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return VarRef(self)

    def __set__(self, instance, value):
        setattr(instance, self.unique_name, Variable(value, self.name))

    @property
    def scad_param(self, instance):
        return self


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


Slice = collections.namedtuple('Slice', 'alpha delta')


def gen_slices(alpha, delta=90):
    beta = alpha
    while beta >= delta:
        yield Slice(alpha - beta, delta)
        beta -= delta
    if beta > 0:
        yield Slice(alpha - beta, beta)


class Vector(collections.namedtuple('Vector', 'x y z')):
    def __new__(cls, x=None, y=None, z=None):
        return super().__new__(cls, x, y, z)

    @property
    def dimensions(self):
        return sum((1 for v in self if v is not None))

    @property
    def xy(self):
        return Vector(self.x, self.y)

    @property
    def is_sparse(self):
        return cor.is_sparse(self, lambda x: x is not None)

    def __repr__(self):
        placeholder = '' if self.is_sparse else None
        values = map(lambda x: placeholder if x is None else str(x), self)
        params = ', '.join(v for v in values if v is not None)
        return '[{}]'.format(params)

    def __add__(self, other):
        def add(a, b):
            if a is None:
                res = b
            elif b is None:
                res = a
            else:
                res = a + b
            return res

        pairs = self._zip(other)
        return Vector(*(add(a, b) for a, b in pairs))

    def __sub__(self, other):
        if not isinstance(other, Vector):
            other = Vector(*other)
        return self + (-other)

    def __neg__(self):
        return Vector(*map(lambda x: None if x is None else -x, self))

    def _zip(self, other):
        if isinstance(other, Iterable):
            return zip(self, other)
        elif isinstance(other, numbers.Number):
            return zip(self, itertools.repeat(other))
        elif isinstance(other, Size):
            return zip(self, other.get_vector(self.dimensions))
        else:
            raise ValueError("Can't zip with {}".format(other))

    def __truediv__(self, other):
        pairs = self._zip(other)
        return Vector(*(a / b for a, b in pairs))

    def __mul__(self, other):
        pairs = self._zip(other)
        def mul(x, y):
            return (x * y if not (x is None or y is None) else None)
        return Vector(*(mul(a, b) for a, b in pairs))

    @property
    def coord(self):
        if self.is_sparse:
            raise RuntimeError("Sparse vector can't be resolved")
        return self[:self.dimensions]

    def updated(self, x=None, y=None, z=None):
        return Vector(
            self.x if x is None else x,
            self.y if y is None else y,
            self.z if z is None else z
        )


class Size:

    def __init__(self, value):
        if isinstance(value, Size):
            value = value.value
        elif not hasattr(value, 'scad_param'):
            value = self._validated_coords(value)
        self.value = value

    def __str__(self):
        if isinstance(self.value, Iterable):
            return str(list(self.value))
        else:
            return str(self.value)

    @property
    def dimensions(self):
        v = self.value
        if isinstance(v, numbers.Number):
            res = 1
        elif isinstance(v, Vector):
            res = v.dimensions
        else:
            res = len(v)
        return res

    def get_vector(self, dimensions):
        v = self.value
        if isinstance(v, list):
            if dimensions != len(v):
                msg = "Requested {} dims, has {}"
                raise ValueError(msg.format(dimensions, len(v)))
            else:
                return v
        else:
            return Vector(itertools.repeat(v, dimensions))

    @classmethod
    def _validated_coord(cls, value):
        if not isinstance(value, numbers.Number):
            raise ValueError('Not a number {}'.format(value))
        return value

    @classmethod
    def _validated_coords(cls, value):
        if isinstance(value, Variable):
            res = value
        if not isinstance(value, Iterable):
            res = cls._validated_coord(value)
        elif isinstance(value, Vector):
            res = value.coord
        elif isinstance(value, str):
                raise ValueError('Number or number seq is expected, got {}'.format(value))
        else:
            res = [cls._validated_coord(v) for v in value]

        return res

    def __add__(self, other):
        if not isinstance(other, Size):
            other = Size(other)

        dims = (self.dimensions, other.dimensions)
        if dims == (1, 1):
            return Size(self.value + other.value)

        max_dim = max(*dims)
        vectors = (x.get_vector(max_dim) for x in (self, other))
        return Size([sum(v) for v in zip(*vectors)])

    def __radd__(self, other):
        return self + other
