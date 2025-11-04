from ase.build import molecule
from tblite.ase import TBLite
atoms = molecule("H2O")
atoms.calc = TBLite(method="GFN2-xTB")
atoms.get_potential_energy()  # in eV
