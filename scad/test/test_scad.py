import unittest
from scad import *


class VectorTests(unittest.TestCase):
    def test_vector_init(self):
        test_data = (
            ({}, '[]', False, 0),
            ({'x': 1}, '[1]', False, 1),
            ({'y': 2}, '[, 2, ]', True, 1),
            ({'z': 3}, '[, , 3]', True, 1),
            ({'x': 1, 'y': 2}, '[1, 2]', False, 2),
            ({'x': 1, 'z': 3}, '[1, , 3]', True, 2),
            ({'y': 2, 'z': 3}, '[, 2, 3]', True, 2),
            ({'x': 1, 'y': 2, 'z': 3}, '[1, 2, 3]', False, 3),
        )
        for kwargs, expected_res, is_sparse, dims in test_data:
            with self.subTest('Init', kwargs=kwargs):
                v = Vector(**kwargs)
                self.assertEqual(repr(v), expected_res)
                self.assertEqual(v.is_sparse, is_sparse)

    def test_vector_add(self):
        V = Vector
        test_data = (
            (V(1), V(2), V(3), 1),
            (V(1), V(y=2), V(1, 2), 2),
            (V(1), V(z=3), V(1, z=3), 2),
            (V(1), V(y=2, z=3), V(1, 2, 3), 3),
            (V(y=2), V(x=1), V(1, 2), 2),
            (V(y=2), V(z=3), V(y=2, z=3), 2),
            (V(1, 2, 3), V(5, 7, 11), V(6, 9, 14), 3),
        )
        for v1, v2, expected_res, dims in test_data:
            with self.subTest('Add', v1=v1, v2=v2):
                res = v1 + v2
                self.assertEqual(res, expected_res)
                self.assertEqual(res.dimensions, dims)

