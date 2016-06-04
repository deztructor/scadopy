def is_sparse(values, pred):
    expect = True
    flip_count = 0
    for v in values:
        if pred(v) != expect:
            flip_count += 1
            if flip_count > 1:
                return True
            expect = not expect
    return False

def compose(*fns):
    l = len(fns)
    if l == 0:
        return lambda: None
    elif l == 1:
        return fns[0]

    def fn(*args, **kwargs):
        rev_fns = fns[-1::-1]
        first, tail = rev_fns[0], rev_fns[1:]
        res = first(*args, **kwargs)
        for fn in tail:
            res = fn(res)
        return res

    return fn


class Member:
    def __init__(self, value=None, name=None, optional=False):
        self._name = name or type(self).__name__
        self._unique_name = '_{}#{}'.format(self.name, id(self))
        self.value = value
        self.optional = optional

    @property
    def name(self):
        return self._name

    def __str__(self):
        return (self.name or '<unknown>') + '=' + (str(self.value) or '<?>')

    def __get__(self, instance, owner):
        attr = self if instance is None \
               else getattr(instance, self._unique_name, self)
        return attr.value

    def __set__(self, instance, value):
        setattr(instance, self._unique_name, Member(value, self.name))


class StructureFactory(type):
    def __new__(cls, name, bases, attrs):
        return super().__new__(cls, name, bases, attrs)

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        for k, v in attrs.items():
            if isinstance(v, Member):
                v._name = k
                v._unique_name = '_{}#{}'.format(cls.__name__, id(v))
                cls._structure_members.add(v.name)
                if v.optional:
                    cls._optional_members.add(v.name)

class Structure(metaclass=StructureFactory):

    @property
    def member_names(self):
        return self._structure_members

    @property
    def optional_members(self):
        return self._optional_members

    def as_dict(self):
        return {name: getattr(self, name)
                for name in self._structure_members}

    _structure_members = set()
    _optional_members = set()

    def __init__(self, **kwargs):
        cls_dict = self.__class__.__dict__
        cls_keys = {name
                    for name, value in cls_dict.items()
                    if isinstance(value, Member)}
        param_keys = set(kwargs.keys())
        if cls_keys != param_keys:
            extra_keys = param_keys - cls_keys
            missing_keys = cls_keys - param_keys - self.optional_members
            if extra_keys or missing_keys:
                err = {
                    'extra': extra_keys,
                    'missing': missing_keys,
                }
                raise ValueError(err)

        for k, v in kwargs.items():
            setattr(self, k, v)
