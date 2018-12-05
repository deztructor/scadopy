import enum
import collections


class Material(enum.Enum):
    PLA = 'pla'
    ABS = 'abs'
    PETG = 'petg'
    PC = 'polycarbonate'
    Nylon = 'nylon'

Filament = collections.namedtuple('Filament', 'perim w')

class Printer(collections.namedtuple('Printer', 'layer material nozzle filament')):

    def __new__(cls, layer_h: float, material: Material, nozzle_d: float = 0.4, filament: Filament = None):
        if layer_h > nozzle_d * 0.75:
            raise Exception("Nozzle is too small for the layer height {}".format(layer_h))

        if filament is None:
            filament = Filament(nozzle_d * 1.125, nozzle_d)

        return super().__new__(cls, layer_h, material, nozzle_d, filament)

    def hole_size(self, size, is_xy):
        return (
            0.9927 * size + 0.3602
            if is_xy
            else 1.0155 * size + 0.2795
        )
