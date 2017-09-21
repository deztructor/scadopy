import enum
import collections


class Material(enum.Enum):
    PLA = 'pla'
    ABS = 'abs'
    PETG = 'petg'
    PC = 'polycarbonate'
    Nylon = 'nylon'


class Printer(collections.namedtuple('Printer', 'layer material nozzle perimeter')):

    def __new__(cls, layer_h, material: Material, nozzle_d=0.4):
        if layer_h > nozzle_d * 0.75:
            raise Exception("Nozzle is too small for the layer height {}".format(layer_h))

        return super().__new__(cls, layer_h, material, nozzle_d, nozzle_d * 0.75)

    def hole_size(self, size, is_xy):
        return (
            0.9927 * size + 0.3602
            if is_xy
            else 1.0155 * size + 0.2795
        )
