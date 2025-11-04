from ase.build import molecule

from xtb_ase import XTB

atoms = molecule("H2O")
atoms.calc = XTB()

atoms.get_potential_energy()

print(atoms.calc.results)
