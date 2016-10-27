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




class Attrs(object):
    def __init__(self, *args, **kwargs):
        self._attrs = {}
        self._attrs.update({k: v for k, v in args})
        self._attrs.update(kwargs)

    def __getattr__(self, name):
        if name == '_attrs' or name.startswith('__'):
            return object.__getattr__(self, name)

        try:
            return self._attrs[name]
        except KeyError as err:
            raise AttributeError(err.args) from err

    def __setattr__(self, name, value):
        if name == '_attrs' or name.startswith('__'):
            object.__setattr__(self, name, value)
        else:
            self._attrs[name] = value

    def as_dict(self):
        return self._attrs

    def get_names(self):
        return self._attrs.keys()

    def as_args(self, names):
        return [self._attrs[k] for k in names]

    @classmethod
    def from_map(cls, names, src):
        return cls(*((k, src.get(k)) for k in names))


