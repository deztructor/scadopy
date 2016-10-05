import math
import collections


class Circle(collections.namedtuple('Circle', 'r')):

    def __new__(cls, r=None, d=None):
        obj = super(Circle, cls).__new__(cls, r)
        if all(v is not None for v in (r, d)):
            raise ValueError('Provide r xor d')

        if r is None:
            if d is None:
                raise ValueError('Need r or d')
            r = d / 2
        return obj

    def chord(self, alpha):
        return 2 * self.r * math.sin(alpha / 2)

    def angle(self, chord):
        return math.asin(chord / (2 * self.r)) * 2

    def chord_from_dcenter(self, dcenter):
        r = self.r
        return math.sqrt(r * r - dcenter * dcenter)

    @classmethod
    def from_chord(cls, chord, dedge):
        sch = chord / 2
        return Circle((dedge + sch * sch / dedge) / 2)
