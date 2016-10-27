import itertools
import math
import unittest

from scad.geom import RightTriangle, Sector


class TriangleTests(unittest.TestCase):
    def test_tri(self):
        a = 0.5
        alpha = math.radians(30)
        beta = math.radians(60)
        c = a / math.sin(alpha)
        b = a / math.tan(alpha)
        vs = dict(locals())
        all_params = {k: vs[k] for k in RightTriangle.params}
        angles = RightTriangle.angles
        sides = RightTriangle.sides
        test_name_sets = (
            itertools.product(angles, sides),
            itertools.permutations(sides, 2),
            itertools.permutations(itertools.chain(angles, sides), 3),
            itertools.permutations(itertools.chain(angles, sides), 4),
            [itertools.chain(angles, sides)],
        )
        for test_names in test_name_sets:
            test_names = list(test_names)
            for names in test_names:
                params = {k: all_params[k] for k in names}
                with self.subTest('Params', params=params):
                    t = RightTriangle(**params)
                    self.assertAlmostEqual(math.hypot(t.a, t.b), t.c)
                    self.assertAlmostEqual(t.alpha + t.beta, math.pi / 2)

class SectorTests(unittest.TestCase):
    def test_create(self):
        alpha = math.radians(30)
        beta = math.radians(60)
        chord = 1
        a = chord / 2
        r = a / math.sin(alpha)
        dcenter = a / math.tan(alpha)
        dedge = r - dcenter
        vs = dict(locals())
        all_params = {k: vs[k] for k in Sector.params}
        angles = Sector.angles
        lines = Sector.lines
        test_name_sets = (
            itertools.product(angles, lines),
            itertools.permutations(lines, 2),
            itertools.permutations(itertools.chain(angles, lines), 3),
            itertools.permutations(itertools.chain(angles, lines), 4),
            [itertools.chain(angles, lines)],
        )
        for test_names in test_name_sets:
            test_names = list(test_names)
            for names in test_names:
                params = {k: all_params[k] for k in names}
                with self.subTest('Params', params=params):
                    s = Sector(**params)
                    self.assertIsInstance(s._t, RightTriangle)
