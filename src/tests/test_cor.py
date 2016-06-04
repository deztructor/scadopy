import cor
import unittest

class TestStructure(cor.Structure):
    a = cor.Member()
    x = cor.Member(optional=True)

    b = cor.Member()
    y = cor.Member('foo', optional=True)


class StructureTests(unittest.TestCase):
    def test_structure(self):
        with self.assertRaises(ValueError):
            obj = TestStructure()

        with self.assertRaises(ValueError):
            obj = TestStructure(a=1, x=3, y=4, z=5)

        obj = TestStructure(a=1, b=2)
        self.assertEqual([obj.a, obj.b, obj.x, obj.y], [1, 2, None, 'foo'])

        obj = TestStructure(a=1, b=2, x='bar')
        self.assertEqual([obj.a, obj.b, obj.x, obj.y], [1, 2, 'bar', 'foo'])

        obj.x = 'foobar'
        self.assertEqual([obj.a, obj.b, obj.x, obj.y], [1, 2, 'foobar', 'foo'])

        self.assertEqual(obj.member_names, {'a', 'b', 'x', 'y'})
        self.assertEqual(obj.optional_members, {'x', 'y'})
        self.assertEqual(
            obj.as_dict(),
            {'a': 1, 'b': 2, 'x': 'foobar', 'y': 'foo'}
        )


if __name__ == '__main__':
    unittest.main()
