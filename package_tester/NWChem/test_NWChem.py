# Example: structure optimization of hydrogen molecule
from ase import Atoms
from ase.optimize import BFGS,LBFGS
from ase.calculators.nwchem import NWChem
from ase.io import write
atoms = Atoms('H2',
           positions=[[0, 0, 0],
                      [0, 0, 0.7]])

## either with plane wave           
atoms.calc = NWChem(xc='PBE')

# or gaussian type
atoms.calc = NWChem(label='calc/nwchem',
              dft=dict(maxiter=2000,
                       xc='B3LYP'),
              basis='6-31+G*')

opt = LBFGS(atoms)
opt.run(fmax=0.02)
write('H2.xyz', atoms)
atoms.get_potential_energy()
