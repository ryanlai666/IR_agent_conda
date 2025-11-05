import numpy as np
import matplotlib.pyplot as plt
import os
# https://ase-lib.org/ase/build/build.html
from ase.build import (
    fcc100, fcc110, fcc111, fcc211,
    bcc100, bcc110, bcc111,
    hcp0001, hcp10m10
)
from ase.build import add_adsorbate, rotate

from ase.collections import g2
from ase.build import molecule

from ase.geometry import get_layers
from ase.calculators.emt import EMT
from ase.calculators.vasp import Vasp
import ase

import shutil
from datetime import datetime


# <<< MODIFIED: Added for calculator type hinting with a fallback
try:
    from ase.calculators.calculator import Calculator
except ImportError:
    Calculator = object # Fallback for type hint

# <<< MODIFIED: Added try/except blocks for new calculators
    # from xtb.ase.calculator import XTB  #  if you use xtb_python as wrapper
try:
    from xtb_ase import XTB, XTBProfile #  if you use xtb_ase as wrapper
    HAS_XTB = True
except ImportError:
    HAS_XTB = False
    
try:
    from tblite.ase import TBLite
    HAS_TBLITE = True
except ImportError:
    HAS_TBLITE = False

try:
    from gpaw import GPAW,PW
    HAS_GPAW = True
except ImportError:
    HAS_GPAW = False

try:
    from ase.calculators.dftb import Dftb
    HAS_DFTBPLUS = True
except ImportError:
    HAS_DFTBPLUS = False

try:
    from ase.calculators.nwchem import NWChem
    HAS_NWCHEM = True
except ImportError:
    HAS_NWCHEM = False

#from ase.optimize import LBFGS
from ase.optimize import BFGS, GPMin, LBFGS, FIRE, BFGSLineSearch, QuasiNewton
from ase.vibrations import Vibrations, Infrared
from ase.vibrations import  VibrationsData
from ase.atoms import Atoms
from ase.constraints import FixAtoms  # <-- Import constraint
from ase.io import write as write # <-- Import for plotting
from langchain_core.tools import tool
#from ase.visualize import view
#from ase.visualize.plot import plot_atoms
#import traceback



# https://ase-lib.org/gettingstarted/surface.html

# Updated internal helper function to select calculator
def _get_calculator(preset: str = 'fast', mode='opt') -> Calculator:
    """
    Internal helper function to select an ASE calculator based on a preset.
    
    This is not a @tool, but is called by other tools.
    
    - 'fast': Uses EMT (fast, mock, always works).
    # <<< MODIFIED: Added xtb and tblite
    - 'xtb': Attempts to use XTB (semi-empirical).
    - 'tblite': Attempts to use TBLite (semi-empirical).
    - 'gpaw': Attempts to use GPAW (DFT).
    - 'default': Attempts to use VASP with medium settings.
    - 'accurate': Attempts to use VASP with high-accuracy settings.
    
    If a requested calculator is not installed,
    it will fall back to EMT with a warning.
    """
    print(f"[Tool._get_calculator] Selecting calculator for preset: '{preset}'")
    
    # Added 'xtb' preset
    if preset == 'xtb':
        if HAS_XTB:
            print("[Tool._get_calculator] Using: XTB (semi-empirical)")
            # GFN2-xTB is a good default method
            #calc = XTB(method="GFN2-xTB", max_iterations=1000)  # not work for perioidc system
            
            # you can also try to do xtb with dftb+
            #from ase.calculators.dftb import Dftb
            #calc = Dftb(Hamiltonian_='xTB',
            #            Hamiltonian_Method="GFN2-xTB",
            #            Driver_='ConjugateGradient',
            #            Driver_MaxForceComponent=1e-4,
            #            Hamiltonian_MaxSCCIterations=500,
            #            kpts=(1, 1, 1)  ## low accary for fast test  
            #      )
            # For more parameter passing of xtb please check:
            # https://andrew-s-rosen.github.io/xtb_ase/reference/xtb_ase/calculator.html#xtb_ase.calculator.XTB
            # https://github.com/grimme-lab/xtb/blob/main/man/xcontrol.7.adoc
            profile = XTBProfile(["xtb", "--opt"])

            #calc = XTB(profile=profile ,method="GFN2-xTB", opt={"maxcycle": 1000, "optlevel": -2 }, scc={"iterations": 300}  )  # not converge for perioidc system
            calc = XTB(profile=profile ,method="GFN2-xTB"  )  # not work for perioidc system
            
        
        else:
            print("[Tool._get_calculator] Warning: 'xtb' preset chosen but 'xtb-python' is not installed.")
            print("[Tool._get_calculator] FALLING BACK TO 'fast' (EMT).")
            calc = EMT()

    # Added 'tblite' preset
    elif preset == 'tblite':  ## currently not used, version contradit to DFTB+
        if HAS_TBLITE:
            print("[Tool._get_calculator] Using: TBLite (semi-empirical)")
            # GFN2-xTB is also a good default for tblite
            calc = TBLite(method="GFN2-xTB")   
        else:
            print("[Tool._get_calculator] Warning: 'tblite' preset chosen but 'tblite-python' is not installed.")
            print("[Tool._get_calculator] FALLING BACK TO 'fast' (EMT).")
            calc = EMT()
            
    elif preset == 'NWChem' or preset=='nwchem':  # still too long for single core laptop, takes more than 1 hrs
        if HAS_NWCHEM:

            print("[Tool._get_calculator] Using: NWChem (gaussian basis functions / plane-waves")
            
            # plane-wave
            # 20.0 Ha is a common starting point for initial tests (~136 eV)
            cutoff_energy_Ha = 10.0 

            calc = NWChem( memory='4096 mb', theory='band',  # Use 'band' for periodic k-point calculations
                xc='LDA',       # Specify the PBE exchange-correlation functional
                # --- CUTOFF ENERGY (Required for plane-wave DFT) ---
                # The 'set' keyword is used to write NWChem RTDB entries. and  PSPW is the module that handles the plane-wave parameters.
                set={
                    'pspw:wcut': cutoff_energy_Ha # Sets the wavefunction cutoff (in Ha)
                },
                
                # --- Other important parameters for periodic systems ---
                kpts=(1, 1, 1)) # K-point sampling

            #calc = NWChem(
            #        
            #        xc='LDA',kpts=(1, 1, 1),theory='dft')  # <-- Add a low cutoff)  # <-- Tell ASE to use a 1x1x1 k-point grid for the slab
            
            if 0:  ## gaussian type
                calc = NWChem(label='calc/nwchem',
                  dft=dict(maxiter=2000,
                           xc='B3LYP'),
                  basis='6-31+G*')

        else:
            print("[Tool._get_calculator] Warning: 'NWChem' preset chosen but 'NWChem' is not installed.")
            print("[Tool._get_calculator] FALLING BACK TO 'fast' (EMT).")
            calc = EMT()

    elif preset == 'gpaw' or preset=='paw':  # still too long for single core laptop, takes more than 1 hrs
        if HAS_GPAW:
            print("[Tool._get_calculator] Using: GPAW (DFT)")
            if mode=='opt':
                calc = GPAW(mode=PW(200), xc='LDA', kpts=(1,1,1))  ## lda for faster calculation and very low encut
                #calc = GPAW(mode='lcao', basis='dzp', xc='LDA', kpts=(1,1,1))  ## lda for faster calculation 
            if mode=='freq':
                calc = GPAW(mode=PW(200),  xc='LDA', kpts=(1,1,1), symmetry='off')  ## lda for faster calculation
                #calc = GPAW(mode='lcao', basis='dzp', xc='LDA', kpts=(1,1,1), symmetry='off')  ## lda for faster calculation

        else:
            print("[Tool._get_calculator] Warning: 'gpaw' preset chosen but 'gpaw' is not installed.")
            print("[Tool._get_calculator] FALLING BACK TO 'fast' (EMT).")
            calc = EMT()
    
    elif preset == 'dftb+':
        if HAS_GPAW:
            print("[Tool._get_calculator] Using: DFTB plus (DFTB)")
            if mode=='opt':
               calc = Dftb(
                label='test',
                Driver_='ConjugateGradient',
                Driver_MaxForceComponent=1e-4,
                kpts=(1, 1, 1), ## low accary for fast test  
                # Au,H,C,N,O,S,P
                Hamiltonian_MaxAngularMomentum_='',
                Hamiltonian_MaxAngularMomentum_O='p',
                Hamiltonian_MaxAngularMomentum_H='s',
                Hamiltonian_MaxAngularMomentum_N='p',
                Hamiltonian_MaxAngularMomentum_C='p',
                Hamiltonian_MaxAngularMomentum_Au='d',
                Hamiltonian_MaxAngularMomentum_S='d',
                Hamiltonian_MaxAngularMomentum_P='d',
                Hamiltonian_MaxSCCIterations=500 
                ) 
            if mode=='freq':  
               calc = Dftb(
                label='test',
                Driver_='ConjugateGradient',
                Driver_MaxForceComponent=1e-6,
                kpts=(1, 1, 1),  # low accracy for fast test
                Hamiltonian_MaxAngularMomentum_='',
                Hamiltonian_MaxAngularMomentum_O='p',
                Hamiltonian_MaxAngularMomentum_H='s',
                Hamiltonian_MaxAngularMomentum_N='p',
                Hamiltonian_MaxAngularMomentum_C='p',
                Hamiltonian_MaxAngularMomentum_Au='d',
                Hamiltonian_MaxAngularMomentum_S='d',
                Hamiltonian_MaxAngularMomentum_P='d',
                Hamiltonian_MaxSCCIterations=1000
                ) 
                #calc = GPAW(mode='lcao', basis='dzp', xc='LDA', kpts=(1,1,1), symmetry='off')  ## lda for faster calculation

        else:
            print("[Tool._get_calculator] Warning: 'gpaw' preset chosen but 'gpaw' is not installed.")
            print("[Tool._get_calculator] FALLING BACK TO 'fast' (EMT).")
            calc = EMT()


    elif preset in ('default', 'accurate'):
        try:
            # Check if VASP is available and environment is set
            if "VASP_COMMAND" not in os.environ or "VASP_PP_PATH" not in os.environ:
                raise EnvironmentError("VASP_COMMAND or VASP_PP_PATH env vars not set.")
            
            if mode=='freq' or mode=='ir':
                if preset == 'accurate':
                    calc = Vasp(prec='Accurate',
                            encut=500,
                            kpts=(8, 8, 2),
                    ediff=1E-8,
                    isym=0,
                    idipol=4,       # calculate the total dipole moment
                    dipol=water.get_center_of_mass(scaled=True),
                    ldipol=True) 
                else:
                    calc = Vasp(prec='Normal',
                    kpts=(2, 2, 1),
                    ediff=1E-6,
                    isym=0,
                    idipol=4,       # calculate the total dipole moment
                    dipol=water.get_center_of_mass(scaled=True),
                    ldipol=True) 

            
            if mode=='opt':
            
                if preset == 'accurate':
                    # High-accuracy VASP (example settings)
                    print("[Tool._get_calculator] Trying: VASP (accurate)")
                    calc = Vasp(
                        xc='PBE',
                        encut=500,      # Higher cutoff for accuracy
                        kpts=(4, 4, 1), # Denser k-point mesh
                        ismear=0,       # Gaussian smearing
                        sigma=0.05,
                        ivdw=12,        # Add vdW correction (e.g., DFT-D3)
                        lreal=False     # Use reciprocal space projection
                    )
                else: # 'default'
                    # Default/medium VASP (example settings)
                    print("[Tool._get_calculator] Trying: VASP (default)")
                    calc = Vasp(
                        xc='PBE',
                        encut=400,
                        kpts=(2, 2, 1),
                        ismear=0,
                        sigma=0.05
                    )
                print("[Tool._get_calculator] VASP calculator selected.")
            
        except (ImportError, EnvironmentError, FileNotFoundError) as e:
            print(f"[Tool._get_calculator] VASP setup failed ({e}).")
            print("[Tool._get_calculator] FALLING BACK TO 'fast' (EMT).")
            print("[Tool._get_calculator] To use VASP, ensure it's installed and VASP_COMMAND/VASP_PP_PATH env vars are set.")
            calc = EMT()
    
    else:
        # Default to 'fast' (EMT)
        if preset != 'fast':
            print(f"[Tool._get_calculator] Warning: Unknown preset '{preset}'. Defaulting to 'fast' (EMT).")
        print("[Tool._get_calculator] Using: EMT (fast)")
        calc = EMT()
        
    return calc
# <<< END MODIFIED SECTION


@tool
def setup_slab_tool(
    surface_metal: str = 'Au',
    surface_metal_facet: str = '111',
    surface_metal_structure_type: str = 'fcc',
    slab_size: tuple = (2, 2, 3), ## smaller cell for faster calcualtion
    preset: str = 'fast' 
) -> Atoms:
    """
    Sets up the initial atomic geometry for a metal slab using ASE.

    This function builds a slab based on the crystal structure (fcc, bcc, hcp)
    and the specified facet.

    Args:
        surface_metal: The chemical symbol for the metal (e.g., 'Au', 'Pt').
        surface_metal_facet: The Miller index of the surface (e.g., '111', '100').
        surface_metal_structure_type: The crystal structure ('fcc', 'bcc', 'hcp').
        slab_size: A tuple defining the size of the supercell (e.g., (3, 3, 3)).

    Returns:
        An ASE Atoms object representing the slab.

    Raises:
        ValueError: If the combination of structure_type and facet is not supported.
    """
    print(f"[Tool] Setting up geometry of {surface_metal}({surface_metal_facet})-{surface_metal_structure_type} {slab_size}...")
    # 1. Define a mapping from structure and facet to the correct ASE function
    
    structure_map = {
        'fcc': {
            '100': fcc100,
            '110': fcc110,
            '111': fcc111,
            '211': fcc211,
        },
        'bcc': {
            '100': bcc100,
            '110': bcc110,
            '111': bcc111,
        },
        'hcp': {
            '0001': hcp0001,
            '10m10': hcp10m10, # Corresponds to (10-10)
            # '1010': hcp10m10 # You could add an alias
        }
    }    
    

    # 2. Try to get the appropriate builder function
    builder_func = structure_map.get(surface_metal_structure_type, {}).get(surface_metal_facet)

    # 3. Build the slab or raise an error
    if builder_func:
        slab = builder_func(surface_metal, size=slab_size, vacuum=10.0)
    else:
        # If the combination was not found, raise a clear error
        raise ValueError(
            f"Unsupported combination: structure_type='{surface_metal_structure_type}' "
            f"with facet='{surface_metal_facet}'. "
            "Supported combinations are: \n"
            f"{list(structure_map.keys())} with facets {structure_map}"
        )


    # 4. Set periodic boundary conditions
    #if preset== 'xtb':
        #print ("\n\n###### for xtb, we turn off PBC ######\n\n turn")
        #slab.pbc = (False, False, False) 
    #else :
        #print ("\n\n###### PBC for slab ######\n\n turn")
        slab.pbc = (True, True, True)

    # 5. Optional: View the slab (uncomment to use)
    #view(slab)


    print(f"[Tool] Geometry setup complete. Total atoms: {len(slab)}")
    return slab

@tool
def setup_adsorbate_tool(
    slab: Atoms,
    adsorbate_formula: str = 'H2O',
    adsorbate_height: float = 2.2
) -> Atoms:
    """
    Places an adsorbate (e.g., water) on top existed metal slab using ASE.
    """    
    
    # 1. Build the adsorbate (water molecule)
    adsorbate = molecule(adsorbate_formula)
    
    slab.pbc = (True, True, True)

    ## rotate the water
    adsorbate.rotate(180, 'x', center='COM')
    #atoms.rotate(180, 'x', center='COM')
    #adsorbate = Atoms(adsorbate_formula, positions=[ (0, 0.76, 0.59),  (0, -0.76, 0.59),(0, 0, 0)])

    # 2. Add the adsorbate to the slab
    add_adsorbate(slab, adsorbate, height=adsorbate_height, position='ontop')

    # 3. fix the bottom layer of the slab
    layers = get_layers(slab, (0, 0, 1), tolerance=0.3)[0]
    bot_indices = np.argwhere((layers == 0) | (layers == 1) ).flatten() # fix bottom 2 layers
    c = FixAtoms(indices=bot_indices)
    slab.set_constraint(c)
    #view(slab)
    return slab




@tool
def setup_molecule_tool(
    adsorbate_formula: str = 'H2O',
) -> Atoms:
    """
    Places an adsorbate (e.g., water) on top existed metal slab using ASE.
    """    
    
    # 1. Build the adsorbate (water molecule)
    #adsorbate = molecule(adsorbate_formula)
    
    mol = molecule(adsorbate_formula, vacuum=8.0) 

    return mol





@tool
def run_opt_tool(
    atoms: Atoms,
    optimizer_fmax: float = 0.05,
    calculator_preset: str = 'fast', # <<< MODIFIED: This argument is now used
    system_name: str='trial'
) -> Atoms: 
    """
    Runs the geometry optimization.
    
    Uses a calculator based on the 'calculator_preset'.
    'fast' = EMT (mock)
    'xtb'/'tblite' = Semi-empirical

    # 'default'/'accurate' = VASP (requires setup) are not not included yet
    """
    print(f"[Tool] Running Optimizaiton with preset: {calculator_preset}...")
    
    # --- Calculator Selection ---
    try:
        atoms.calc = _get_calculator(calculator_preset)
    except Exception as e:
        print(f"[Tool] Critical error setting calculator: {e}. Defaulting to EMT.")
        atoms.calc = EMT()
    # ---

    # 1. Geometry Optimization
    #optimizer = LBFGS(atoms, logfile=None)
    #qn_opt = BFGSLineSearch(atoms, trajectory='results_qn.traj', logfile='results_qn.log')
    #qn_opt.run(fmax=maxf)
    
    optimizer = BFGSLineSearch(atoms,  trajectory='results_'+system_name+'_opt.traj',logfile='results_'+system_name+'_opt.log') # first pre-optimzation are not saved
    optimizer.run(fmax=optimizer_fmax)
    print(f"[Tool] Optimization complete.")
    
    return atoms 


@tool
def run_opt_freq_tool(
    atoms: Atoms,
    optimizer_fmax: float = 0.01, ## higher force requirment
    vib_delta: float = 0.01,
    calculator_preset: str = 'fast', # <<< MODIFIED: This argument is now used
    adsorbate_formula: str = 'H2O',
    system_name: str='trial'

) -> tuple[ Atoms,Vibrations]:    
    """
    Runs the (mock) electronic structure calculation.
    
    This tool performs two steps:
    1. Geometry Optimization: Relaxes the atoms to their nearest energy minimum.
    2. Vibrational Analysis: Computes vibrational frequencies of the adsorbate.
       <<< MODIFIED: This now uses the Infrared class to get intensities. >>>

    Uses a calculator based on the 'calculator_preset'.
    'fast' = EMT (mock)
    'xtb'/'tblite' = Semi-empirical
    'default'/'accurate' = VASP (requires setup)
    """    

    print(f"[Tool] Running Opt/Freq with preset: {calculator_preset}...")    

    # --- Calculator Selection ---
    try:
        atoms.calc = _get_calculator(calculator_preset, mode='freq')  ## select freq mode and turn off symmetry
        
        # <<< MODIFIED: Set DIPOL tag here, where 'atoms' object exists
        # This fixes a bug in the original _get_calculator function
        if calculator_preset in ('default', 'accurate') and atoms.calc.name == 'vasp':
            print("[Tool] Setting VASP DIPOL tag...")
            atoms.calc.set(dipol=atoms.get_center_of_mass(scaled=True))
            
    except Exception as e:
        print(f"[Tool] Critical error setting calculator: {e}. Defaulting to EMT.")
        atoms.calc = EMT()
    # ---

    # 1. Geometry Optimization
    optimizer = LBFGS(atoms, logfile='results_'+system_name+'_opt_freq.log')
    #optimizer = BFGS(atoms, logfile=False)
    optimizer.run(fmax=optimizer_fmax)

    #print(f"[Tool] Optimization complete.")

    # 2. Vibrational Analysis
    # We select only the water atoms (last 3) to calculate vibrations.  
    # This is a common approximation to save computation.


    adsorbate_atom_num = len(molecule(adsorbate_formula))
    water_indices = list(range(len(atoms) - adsorbate_atom_num, len(atoms)))   # this number depends on the adsorbate.
    
    # <<< MODIFIED: Use Infrared instead of Vibrations to get intensities
    
   
   ## will remove each vibration information of last calculations for now
    vib_name = 'vib' # This is the default name
    if os.path.exists(vib_name):
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_name = f"{vib_name}_archive_{timestamp}"
        
        print(f"[Tool] Stale directory '{vib_name}' found. Renaming to '{archive_name}'.")
        os.rename(vib_name, archive_name)


    # the ASE implement IR rquire dipole moment calucaltions
    # https://ase-lib.org/ase/vibrations/infrared.html
    #The calculator object (calc) linked to the Atoms object (atoms) must have the attribute:
    if atoms.calc.name == 'vasp' or atoms.calc.name == 'gpaw':  
        print("[Tool] Initializing Infrared calculation...")
        
        
        #https://gitlab.com/ase/ase/-/blob/master/doc/ase/vibrations/H2_ir.py?ref_type=heads
        # finite displacement for vibrations
        if atoms.calc.name == 'gpaw': 
            atoms.calc.set(symmetry={'point_group': False})
        
        ir_calc = Infrared(atoms, indices=water_indices, delta=vib_delta)

        ir_calc.run()
        
        print(f"[Tool] Infrared analysis complete.")
        ## start to print the summary
        print("--- Infrared Summary ---")
        ir_calc.summary
        print("------------------------")
        
        #vib_data = VibrationsData(atoms, hessian, VibrationsData.indices_from_mask(water_indices))
        #print(vib_data)
        # Get the VibrationsData object which stores the results
        vib_data = ir_calc.get_vibrations()
        # <<< MODIFIED: Return the Infrared object
        return atoms, ir_calc
    
    # No dipole moment, directly use vibration to do the boarding   both EMT, and dftb doesn't have dipole calc.get_dipole_moment() available
    # therefore, we consdier vibration mode as IR
    else:
        vib = Vibrations(atoms, indices=water_indices, delta=vib_delta)
        vib.run()

        print(f"[Tool] Vibrational analysis complete.")
        ## start to print the summary
        vib.summary
        
        
        #vib_data = VibrationsData(atoms, hessian, VibrationsData.indices_from_mask(water_indices))
        #print(vib_data)
        # Get the VibrationsData object which stores the results
        if 0:  ## if need hessian
            vib_data = vib.get_vibrations()

            # Extract the Hessian matrix
            # get_hessian() returns a (N, 3, N, 3) array
            hessian_4d = vib_data.get_hessian()
            
            # get_hessian_2d() returns a (3N, 3N) array, which is often more useful
            hessian_2d = vib_data.get_hessian_2d()
            print("hessian_2d:\n",hessian_2d )
        
        #hessian = atoms.calc.get_hessian(atoms=atoms)

        #print(hessian)
        return atoms, vib
    


    if 0: # <-- Old code block, disabled

        adsorbate_atom_num = len(molecule(adsorbate_formula))
        water_indices = list(range(len(atoms) - adsorbate_atom_num, len(atoms)))   # this number depends on the adsorbate.
        
        print(range(len(atoms) - adsorbate_atom_num, len(atoms)))
        print(adsorbate_atom_num)
        vib = Vibrations(atoms, indices=water_indices, delta=vib_delta, nfree=2)
        #vib = Vibrations(atoms)

        # Set this NEW calculator on the atoms object  that the Vibrations class will use for its calculations.
        print("Starting vib.run()...")
        vib.run()
        print("vib.run() complete.")  
        #vib.run()
        #print(vib.summary())
        N_mask = atoms.get_chemical_symbols() == 'N'
        vib_data = VibrationsData(atoms, hessian, VibrationsData.indices_from_mask(N_mask))
        print(vib_data)
        #vib.summary(method='frederiksen')
        #print(vib)
        print(f"[Tool] Vibrational analysis complete.")
        return atoms , vib


@tool
def extract_and_plot_spectra_tool(
    vib_run: Vibrations,
    run_label: str,
    color: str,
    ax: plt.Axes,
    broaden_width_cm: float = 15.0
) -> dict:
    """
    Extracts frequencies from a Vibrations object and plots a broadened
    IR spectrum on a given matplotlib axis.
    """
    print(f"[Tool] Extracting and plotting spectrum for run: {run_label}...")
    
    # 1. Get frequencies (in meV). ASE Vibrations gives complex numbers.
    # The real part is the frequency. We convert from meV to cm^-1.
    # 1 meV = 8.0655 cm^-1
    frequencies_mev = vib_run.get_frequencies().real
    frequencies_cm = frequencies_mev * 8.0655
    
    # Filter out non-physical (negative/zero) frequencies
    frequencies_cm = frequencies_cm[frequencies_cm > 50] # Ignore low-freq noise
    
    # --- MOCK INTENSITIES ---
    # EMT/XTB/TBLite calculators do not support dipole moments by default, 
    # so we cannot get real IR intensities. We will mock them.
    # A real VASP calculation would need 'LEPSILON = .TRUE.' to get intensities.
    intensities = np.ones_like(frequencies_cm)
    
    # 2. Broaden the spectrum (simulate a real spectrometer)
    def gaussian(x, center, width):
        return (1.0 / (width * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - center) / width)**2)

    # Create x-axis for the plot
    x_spectrum = np.linspace(min(frequencies_cm) - 200, max(frequencies_cm) + 200, 1000)
    y_spectrum = np.zeros_like(x_spectrum)
    
    # Add a Gaussian for each peak
    for freq, intensity in zip(frequencies_cm, intensities):
        y_spectrum += intensity * gaussian(x_spectrum, freq, broaden_width_cm)
        
    # 3. Plot on the provided axis
    ax.plot(x_spectrum, y_spectrum, label=f"{run_label} (delta={vib_run.delta:.4f})", color=color)
    
    print(f"[Tool] Notice:! Due to lack of dipole moment in DFTB+ and EMT calculator, we consider all mode to be IR active" )
    print(f"[Tool] Plotting complete for {run_label}.")
    
    return {
        "frequencies_cm": frequencies_cm,
        "intensities": intensities,
        "spectrum_x": x_spectrum,
        "spectrum_y": y_spectrum,
        "label": run_label
    }

@tool
def plot_structure_comparison_tool(
    atoms_1: Atoms,
    atoms_2: Atoms,
    atoms_3: Atoms,
    atoms_4: Atoms,
    filename: str = "structure_comparison.png"
) -> str:
    """
    Saves a 1x4 subplot comparison of four ASE Atoms structures, 
    visualizing constraints.
    
    Uses a top-down view ('-90x').
    """
    #print(f"[Tool] Plotting structure comparison to {filename}...")
    
    try:
        # 1. Create the subplot figure
        # We create the figure first, then plot directly onto each axis
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        axes = axes.flatten()
        
        # 2. Package atoms objects and their titles
        structures = [atoms_1, atoms_2, atoms_3, atoms_4,atoms_1, atoms_2, atoms_3, atoms_4]
        titles = [
            "Before Opt Metal",
            "After Opt Metal ",
            "Opt Metal (fixed bot 2 layers) + Unoptimized Adsorbate",
            "Opt (Metal (fixed bot 2 layers) + Adsorbate)",
            "top view",
            "top view",
            "top view",
            "top view"
        ]
        
        # 3. Define common plotting arguments
        # This is where we pass the key arguments
        plot_kwargs1 = {
            'rotation': '-90x',      # Top-down view from your original code
            #'show_constraints': True  # currently ase not support enables constraint plotting in plot_atoms function but in view function it works
        }

        plot_kwargs2 = {
            'rotation': ''      # Top-down view from your original code
            #'show_constraints': True  # currently ase not support enables constraint plotting in plot_atoms function but in view function it works
        }
        # 4. Loop and plot directly onto each subplot axis
        for ax, atoms, title in zip(axes, structures, titles):
            # Use ase.visualize.plot.plot_atoms to draw on the axis
            if title=="top view":
                plot_atoms(
                    atoms,
                    ax,
                    **plot_kwargs2
                )
            else:
                plot_atoms(
                        atoms,
                        ax,
                        **plot_kwargs1
                    )
            ax.set_title(title, fontsize=12)
            ax.set_axis_off() # Turn off the black box border

        # 5. Set overall figure properties and save
        fig.suptitle("Top-Down View of Adsorption Site", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, dpi=300) # Save with high resolution
        
        print(f"[Tool] Successfully saved comparison plot to {filename}")
        
    except Exception as e:
        print(f"[Tool] Error plotting structure comparison: {e}")
        # Print full error for debugging
        traceback.print_exc()
        return f"Error: {e}"
    
    # No 'finally' block needed, as we no longer create temp files
    return f"Plot saved to {filename}"
