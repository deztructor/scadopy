#!/usr/bin/env python3

from collections import namedtuple, Iterable
from enum import Enum
import functools
import itertools
import numbers
import cor
import inspect
import math


class _Bool(namedtuple('Bool', 'value')):
    def __str__(self):
        return 'true' if self.value else 'false'


class ParamAt(Enum):
    Call = 0
    Decl = 1


def format_params_dict(src, param_at, **spec_vars):
    def format_param(name, value):
        fmt = '{name}' if value is None else '{name} = {value}'
        if hasattr(value, 'scad_param'):
            res = fmt.format(name=name, value=value.scad_param)
        if isinstance(value, bool):
            res = fmt.format(name=name, value=_Bool(value))
        elif isinstance(value, str):
            # TODO what about module?
            if value is not None:
                fmt = '{name} = "{value:s}"'
            res = fmt.format(name=name, value=value)
        elif isinstance(value, Vector):
            res = fmt.format(name=name, value=value)
        elif isinstance(value, list):
            res = fmt.format(name=name, value=value)
        elif isinstance(value, Iterable):
            res = fmt.format(name=name, value=list(value))
        else:
            res = fmt.format(name=name, value=value)
        return res

    items = src.items()
    if param_at == ParamAt.Call:
        items = ((name, value)
                 for name, value in items
                 if value is not None)

    formatted_params = (format_param(name, value) for name, value in items)
    formatted_spec = (str(svar(k, v)) for k, v in spec_vars.items())

    return ', '.join(itertools.chain(formatted_params, formatted_spec))


def format_params(src, param_at, **spec_vars):
    if hasattr(src, '_asdict'):
        src = src._asdict()

    return format_params_dict(src, param_at, **spec_vars)


def format_call_params(src, **spec_vars):
    return format_params(src, ParamAt.Call, **spec_vars)


def format_decl_params(src, **spec_vars):
    return format_params(src, ParamAt.Decl, **spec_vars)


def assert_conflicting_args(*args):
    if len([arg for arg in args if arg is not None]) > 1:
        raise ValueError('Conflicting arguments', args)


def assert_passed_together(*args):
    nones_len = len([args for arg in args if arg is None])
    if nones_len not in (0, len(args)):
        raise ValueError('Should be passed together')


class _ProgramMixin:
    def dumps(self, indent=' ' * 4):
        return '\n'.join(self.lines(indent))


class _OneLinerMixin(_ProgramMixin):
    def lines(self, indent, level=0):
        yield (indent * level) + str(self)

class _OneLinerObjectMixin(_ProgramMixin):
    def lines(self, indent, level=0):
        yield (indent * level) + str(self) + ';'


class _TransformFormatMixin(_ProgramMixin):
    def lines(self, indent, level=0):
        yield (indent * level) + str(self)
        yield from self._target.lines(indent, level + 1)


class _CSGMixin:
    def __add__(self, other):
        objects = itertools.chain((self,), other.objects) \
                  if isinstance(other, Union) \
                     else (self, other)
        return Union(*objects)

    def __sub__(self, other):
        objs = (self, other)
        return Difference(*objs)

    def __and__(self, other):
        objs = (self, other)
        return Intersection(*objs)


class Item:
    pass


class Block(Item):
    def __init__(self, items=None):
        super().__init__()
        self._items = items or []

    def __lshift__(self, item):
        return self.append(item)

    def append(self, item):
        if isinstance(item, str):
            self._items.extend(item.split('\n'))
        if isinstance(item, Iterable):
            self._items.extend(item)
        else:
            self._items.append(item)
        return self


class Objects(Block, _ProgramMixin):

    def __init__(self, name, objects):
        if len(objects) > 0:
            dim = objects[0].dimensions
            for obj in objects[1:]:
                if obj.dimensions != dim:
                    msg = "Can't combine objects with different dims: {}, {}"
                    raise ValueError(msg.format(self, obj))
        else:
            dim = None

        super().__init__(objects)
        self._name = name
        self._dimensions = dim

    @property
    def dimensions(self):
        return self._dimensions

    def lines(self, indent, level=0):
        my_indent = indent * level
        yield '{}{}() {{'.format(my_indent, self._name)
        for obj in self._items:
            yield from obj.lines(indent, level + 1)
        yield my_indent + '};'

    @property
    def objects(self):
        return self._items


class Union(Objects, _CSGMixin):
    def __init__(self, *objects):
        return super().__init__('union', objects)

    def __add__(self, other):
        right_objects = (other.objects if isinstance(other, Union) else (other,))
        objects = itertools.chain(self.objects, right_objects)
        return Union(*objects)


class Difference(Objects, _CSGMixin):
    def __init__(self, *objects):
        return super().__init__('difference', objects)

    def __sub__(self, other):
        return Difference(*itertools.chain(self.objects, (other,)))


class Intersection(Objects, _CSGMixin):
    def __init__(self, *objects):
        return super().__init__('intersection', objects)

    def __and__(self, other):
        objs = itertools.chain(self.objects + (other,))
        return Intersection(*objs)


class Vector(namedtuple('Vector', 'x y z')):
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
                res = None
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
        return Vector(*(a * b for a, b in pairs))

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

class Transform(Item, _TransformFormatMixin, _CSGMixin):
    def __init__(self, name, target, data):
        self._name = name
        self._target = target
        self._data = data

    @property
    def dimensions(self):
        return self._target.dimensions

    def __str__(self):
        return '{}({})'.format(
            self._name,
            format_call_params(self._data) if self._data else ''
        )


class _TransformNode:
    def __init__(self, transform_cls, data=None):
        self._cls = transform_cls
        self._data = data

    def __rmul__(self, target):
        return self._cls(target, self._data) if self._data \
            else self._cls(target)


RotateData = namedtuple('RotateData', 'a v')

class _Rotate(Transform):
    def __init__(self, target, data):
        super().__init__('rotate', target, data)

def rotate(angle, vector=None):
    return _TransformNode(_Rotate, RotateData(angle, vector))


VectorParam = namedtuple('VectorParam', 'v')

class _Translate(Transform):
    def __init__(self, target, data):
        super().__init__('translate', target, data)

def translate(vector=None, x=None, y=None, z=None):
    return _TransformNode(_Translate, VectorParam(vector))


class _Scale(Transform):
    def __init__(self, target, data):
        super().__init__('scale', target, data)


def scale(vector):
    return _TransformNode(_Scale, VectorParam(vector))


class _Mirror(Transform):
    def __init__(self, target, data):
        super().__init__('mirror', target, data)


def mirror(vector):
    return _TransformNode(_Mirror, VectorParam(vector))


ResizeParams = namedtuple('ResizeParams', 'newsize auto')

class _Resize(Transform):
    def __init__(self, target, data):
        super().__init__('resize', target, data)


def resize(newsize, auto=None):
    return _TransformNode(_Resize, ResizeParams(newsize, auto))

class LimitedDimTransform(Transform):
    def __init__(self, dimensions, name, target, data=None):
        if target.dimensions not in dimensions:
            raise ValueError("Can transform only object with {} dims".format(dimensions))
        super().__init__(name, target, data)

class _Extrude(LimitedDimTransform):
    def __init__(self, name, target, data=None):
        super().__init__((2,), name, target, data)

    @property
    def dimensions(self):
        return 3


LinearExtrudeParams = namedtuple(
    'LinearExtrudeParams',
    'height center convexity twist slices scale'
)


class _LinearExtrude(_Extrude):
    def __init__(self, target, data):
        super().__init__('linear_extrude', target, data)


def linear_extrude(height=None, center=None, convexity=None,
                   twist=None, slices=None, scale=None):
    data = LinearExtrudeParams(height, center, convexity, twist, slices, scale)
    return _TransformNode(_LinearExtrude, data)


class _RotateExtrude(_Extrude):
    def __init__(self, target):
        super().__init__('rotate_extrude', target)


def rotate_extrude():
    return _TransformNode(_RotateExtrude)

class OffsetParams(namedtuple('OffsetParams', 'r delta chamfer')):
    def __new__(cls, r=None, delta=None, chamfer=None):
        assert_conflicting_args(r, delta)
        assert_passed_together(delta, chamfer)
        return super().__new__(cls, r, delta, chamfer)


class _Offset(LimitedDimTransform):
    def __init__(self, target, data):
        super().__init__((2,), 'offset', target, data)

    @property
    def dimensions(self):
        return 2

def offset(r=None, delta=None, chamfer=None):
    return _TransformNode(_Offset, OffsetParams(r, delta, chamfer))

class _Minkowski(Transform):
    def __init__(self, target):
        super().__init__('minkowsi', target, None)

def minkowsi():
    return _TransformNode(_Minkowski)

class Geometry(Item, _CSGMixin):
    def __init__(self, dimensions, name, data, **special_vars):
        self._dimensions = dimensions
        self._data = data
        self._name = name
        self._special_vars = special_vars

    @property
    def dimensions(self):
        return self._dimensions

    def __str__(self):
        params = format_call_params(self._data, **self._special_vars)
        return '{}({});'.format(self._name, params)


PolygonData = namedtuple('PolygonData', 'points paths convexity')

class Polygon(Geometry, _OneLinerObjectMixin):
    def __init__(self, points, paths=None, convexity=None):
        super().__init__(2, 'polygon', PolygonData(points, paths, convexity))


PolyhedronData = namedtuple('PolyhedronData', 'points triangles convexity')

class Polyhedron(Geometry, _OneLinerObjectMixin):
    def __init__(self, points, triangles=None, convexity=None):
        data = PolyhedronData(points, triangles, convexity)
        super().__init__(3, 'polyhedron', data)


RectData = namedtuple('RectData', 'size center')

class RectGeometry(Geometry, _OneLinerObjectMixin):
    def __init__(self, dimensions, name, size, center=None):
        super().__init__(dimensions, name, RectData(Size(size), center))

    @property
    def size(self):
        return self._data.size.get_vector(self.dimensions)

    def expanded(self, delta):
        size = self._data.size
        delta = [x * 2 for x in delta]
        return type(self)(
            size = size + delta,
            center = self._data.center)


class Square(RectGeometry):
    def __init__(self, size, center=None):
        super().__init__(dimensions=2, name='square', size=size, center=center)

    @property
    def points(self):
        data = self._data
        w, h = self.size
        res = [[0, 0], [w, 0], [w, h], [h, 0]]
        if data.center:
            hw, hh = w/2, h/2
            res = [[x-hw, y-hh] for x, y in res]
        return Polygon(res)


def square(size, **kwargs):
    return Square(size, **kwargs)


class Cube(RectGeometry):
    def __init__(self, size, center=None):
        super().__init__(dimensions=3, name='cube', size=size, center=center)


class CircleData(namedtuple('CircleData', 'r d')):
    def __new__(cls, r, d):
        assert_conflicting_args(r, d)
        return super().__new__(cls, r, d)

class RoundGeometry(Geometry, _OneLinerObjectMixin):
    def __init__(self, dimensions, name, r, d, **special_vars):
        super().__init__(dimensions, name, CircleData(r, d), **special_vars)


class Sphere(RoundGeometry):
    def __init__(self, r=None, d=None):
        super().__init__(dimensions=3, name='sphere', r=r, d=d)


class Circle(RoundGeometry):
    def __init__(self, r=None, d=None, **special_vars):
        super().__init__(dimensions=2, name='circle', r=r, d=d, **special_vars)


def circle(**kwargs):
    return Circle(**kwargs)


class CylinderParams(namedtuple('CylinderParams', 'h r d r1 r2 d1 d2 center')):
    def __new__(cls, h, r, d, r1, r2, d1, d2, center):
        for args in ((r, r1, r2), (d, d1, d2), (r, d), (r1, d1), (r2, d2)):
            assert_conflicting_args(*args)
        return super().__new__(cls, h, r, d, r1, r2, d1, d2, center)


class Cylinder(Geometry, _OneLinerObjectMixin):
    def __init__(self, h, r=None, d=None, r1=None, r2=None,
                 d1=None, d2=None, center=None, **special_vars):
        params = CylinderParams(h, r, d, r1, r2, d1, d2, center)
        super().__init__(dimensions=3, name='cylinder', data=params,
                         **special_vars)


def cylinder(h, **kwargs):
    return Cylinder(h, **kwargs)


ImportParams = namedtuple('ImportParams', 'file convexity')


class Import(Geometry, _OneLinerObjectMixin):
    def __init__(self, file, convexity=None, **special_vars):
        params = ImportParams(file, convexity)
        super().__init__(dimensions=2, name='import', data=params,
                         **special_vars)


class SpecialVariable(namedtuple('SpecialVariable', 'name value'),
                      _OneLinerObjectMixin):

    def __str__(self):
        return '${} = {}'.format(self.name, self.value)


def svar(name, value):
    return SpecialVariable(name, value)


class StatementMixin:

    def __add__(self, other):
        return Expression([self, '+', other])

    def __radd__(self, other):
        return Expression([other, '+', self])

    def __sub__(self, other):
        return Expression([self, '-', other])

    def __rsub__(self, other):
        return Expression([other, '-', self])

    def __mul__(self, other):
        return Expression([self, '*', other])

    def __rmul__(self, other):
        return Expression([other, '*', self])

    def __truediv__(self, other):
        return Expression([self, '/', other])

    def __rtruediv__(self, other):
        return Expression([other, '/', self])


class FnExpression(StatementMixin, _OneLinerMixin):
    def __init__(self, name, var):
        self._name = name
        self._var = var

    def __str__(self):
        return '{}({})'.format(self._name, self._var)


def _mk_math_fn(name):
    def fn(x):
        return FnExpression(name, x) \
            if isinstance(x, StatementMixin) \
            else getattr(math, name)(x)
    return fn


sin = _mk_math_fn('sin')
cos = _mk_math_fn('cos')


class Expression(StatementMixin, _OneLinerMixin):
    def __init__(self, statements):
        self._statements = statements

    def __str__(self):
        return '({})'.format(self._format_statements())

    def _format_statements(self):
        return ' '.join(self._format_statement(x) for x in self._statements)

    def _format_statement(self, s):
        if isinstance(s, str):
            return s
        return str(s)


def avar(var_name=None, **kwargs):
    if var_name:
        return AVariable(var_name)
    elif len(kwargs) == 1:
        name, value = tuple(kwargs.items())[0]
        return AVariable(name, value)
    else:
        raise ValueError('Should be single k=v, got {}'.format(kwargs))


class AVariable(StatementMixin):
    def __init__(self, name, value=None):
        self._name = name
        self._value = value

    def __str__(self):
        return self._name if self._value is None \
            else '{} = {}'.format(self._name, self._value)


class Scene(Block):
    def __init__(self, **special_vars):
        super().__init__()
        self._special_vars = special_vars
        self._indent = ' ' * 4

    def __enter__(self, indent=' ' * 4):
        self._indent = indent
        return self

    def __exit__(self, exc, exc_type, tb):
        if exc is None:
            print(self.dumps())

    def lines(self, indent, level=0):
        special_lines = (svar(name, value).lines(indent, level)
                         for name, value in self._special_vars.items())
        items_lines = (x.lines(indent, level) for x in self._items)
        return itertools.chain(*special_lines, *items_lines)

    def dumps(self, indent=' ' * 4):
        return '\n'.join(self.lines(indent))

    def dump(self, stream, indent=' ' * 4):
        return stream.writelines('\n' + line for line in self.lines(indent))


class VarRef(StatementMixin):
    def __init__(self, var):
        self._var = var

    def __str__(self):
        return self._var.name


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


class ScopeFactory(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        for k, v in attrs.items():
            if isinstance(v, Variable):
                v.name = k
                v.unique_name = '_{}#{}'.format(cls.__name__, v._counter)
                v._counter += 1


class Scope(metaclass=ScopeFactory):
    def __init__(self, **kwargs):
        self._items = []

        cls_dict = self.__class__.__dict__
        for k, v in kwargs.items():
            if not isinstance(cls_dict.get(k, None), Variable):
                raise ValueError('No such scope variable: {}'.format(k))
            setattr(self, k, v)

    @property
    def variables(self):
        var_or_none = lambda v: v if isinstance(v, Variable) else None

        cls_dict = self.__class__.__dict__
        obj_dict = self.__dict__

        obj_variables = [v for v in map(var_or_none, obj_dict.values()) if v is not None]

        obj_names = set(v.name for v in obj_variables)
        cls_names = set(cls_dict.keys()) - obj_names

        normalized_cls_variables = map(lambda k: var_or_none(cls_dict[k]), cls_names)
        cls_variables = (v for v in normalized_cls_variables if v is not None)
        return dict((v.name, v.value) for v in itertools.chain(obj_variables, cls_variables))

    def _lines(self, lines, indent, level):
        open_fmt = indent * level + '{scope_type} {name}({params}) {{'

        block_open = open_fmt.format(
            scope_type=self._type_name,
            name=self.name,
            params=format_decl_params(self.variables)
        )
        block_close = indent * level + '}'
        return itertools.chain([block_open], lines, [block_close])


class ModuleInstance(_OneLinerObjectMixin):
    def __init__(self, module):
        self._module = module

    def __repr__(self):
        return self._module.call_repr

    @property
    def dimensions(self):
        return self._module.dimensions


class _Repeat(Scope):
    def __init__(self, counter_name, counter, *args):
        super().__init__(**kwargs)
        self._type_name = 'for'
        self.name = ''


class Module(Scope):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type_name = 'module'
        self._declaration = None

    def __call__(self, **kwargs):
        call_keys = set(kwargs.keys())
        vars = self.variables
        var_keys = set(vars.keys())
        extra = call_keys - var_keys
        if extra:
            raise ValueError('Extra params: {}'.format(extra))
        return ModuleInstance(type(self)(**kwargs))

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def call_repr(self):
        return '{}({})'.format(self.name, self._variables_csv)

    @property
    def _variables_csv(self):
        def fmt_kv(k, v):
            fmt = '{k}={v}' if v is not None else '{k}'
            return fmt.format(k=k, v=v)

        return ', '.join(fmt_kv(k, v) for k, v in self.variables.items())

    def __repr__(self):
        lines = ['module {} {{'.format(self.call_repr), self.declaration(), '}']
        return '\n'.join(lines)

    @property
    def declaration(self):
        if self._declaration is None:
            self._declaration = self.get_object()
        return self._declaration

    @property
    def definition(self):
        declaration = self.declaration
        if declaration is None:
            return Union(*self._items)
        elif len(self._items) == 0:
            return declaration
        else:
            return Union(declaration, *self._items)

    @property
    def dimensions(self):
        return self.definition.dimensions

    def lines(self, indent, level=0):
        item_lines = self.definition.lines(indent, level+1)
        return self._lines(item_lines, indent, level)

    def dumps(self, indent=' ' * 4):
        return '\n'.join(self.lines(indent))


def module(fn):
    sig = inspect.signature(fn)
    def get_object(obj):
        params = {name: getattr(obj, name) for name in sig.parameters}
        return fn(**params)

    vars = {name: Variable() for name in sig.parameters}
    cls = ScopeFactory(fn.__name__, (Module,), vars)
    cls.get_object = get_object

    obj = cls()
    return obj


# for (x = [a:b:c]) {
# }
def for_each(fn):
    sig = inspect.signature(fn)
    vars = {name: Variable() for name in sig.parameters}

    def wrapper(**kwargs):
        return for_each_loop()


def expand_z(xy, z):
    def expand(xy, z):
        if not isinstance(xy, Iterable) or isinstance(xy, str):
            raise ValueError('Expected sequence for {}'.format(xy))
        xy = tuple(xy)
        if len(xy) != 2:
            raise ValueError('Expected 2 dims, got {}'.format(len(xy)))

        return Vector(*xy, z)

    return Size(expand(xy.value.get_vector(2), z)) if isinstance(xy, Size) \
        else expand(xy, z)

def dump(obj, fname):
    with open(fname, 'w') as fp:
        scene = Scene()
        scene.append(obj).dump(fp)

if __name__ == '__main__':
    print(Size(1))
    print(Size([1, 2, 3.1]))
    print(_Bool(True))
    print(Cube([1, 2, 3]), Cube(4.5, True))
    print(Sphere(r=1.2))
    u = Sphere(r=1.2) + Cube(1)
    print(u)
    print(u.dumps())
    u2 = u + u
    print(u2.dumps())
    d = u - Cube(2) - Sphere(r=6)
    print(d.dumps())

    i = u & d
    print(i.dumps())
    obj2d = Square(2) + Circle(r=3)
    print(obj2d.dumps())
    extrude = obj2d * linear_extrude(height=10, twist=2)
    print(extrude.dumps())
    print((extrude + u).dumps())
    sph = Sphere(d=2) * rotate(90) * scale([2, 2, 1]) * resize([10, 20, 30]) \
          * mirror([2, 3, 4])
    u = sph + (Square(4) - Square(3)) * linear_extrude(height=4) + \
        Circle(r=2) * rotate_extrude()
    print(u.dumps())
    u2 = Union(sph, u)
    print(u2.dumps())
    print((Square(3) * offset(r=3)).dumps())
    print(Cylinder(h=2).dumps())
    print(Square([2, 3], True).points)
    with Scene() as scene:
        scene << Square(2) << Square([3, 4]) * linear_extrude(height=5)
        scene << Polygon([[3, 4, 5], [6, 7, 8]]) + Square(4)
    print(Size(1) + Size(2))
    print(Size(1) + Size([1, 2, 3]))
    print(Size([5, 5, 5]) + Size([1, 2, 3]))
    print([3, 4, 5] + Size(4))
