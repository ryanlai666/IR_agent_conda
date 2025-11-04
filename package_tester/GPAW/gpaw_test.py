from gpaw import GPAW
from ase import Atoms
atoms = Atoms('N2', positions=[[0, 0, -1], [0, 0, 1]])


calc = GPAW(mode='lcao', basis='dzp', txt='gpaw.txt', xc='LDA')

atoms.calc = calc
atoms.center(vacuum=3.0)
print(atoms)
e = atoms.get_potential_energy()
print('Energy', e)
f = atoms.get_forces()
print('Forces')
print(f)
