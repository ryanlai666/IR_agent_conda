import numpy as np
import matplotlib.pyplot as plt
import traceback
from ase.atoms import Atoms
from ase.vibrations import Vibrations
from ase.visualize.plot import plot_atoms
from langchain_core.tools import tool


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
    # Handle case with no valid frequencies
    if len(frequencies_cm) == 0:
        print("[Tool] Warning: No valid frequencies found > 50 cm^-1.")
        x_spectrum = np.linspace(0, 4000, 1000)
        y_spectrum = np.zeros_like(x_spectrum)
    else:
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
    Saves a 2x4 subplot comparison of four ASE Atoms structures, 
    visualizing constraints with side and top-down views.
    """
    #print(f"[Tool] Plotting structure comparison to {filename}...")
    
    try:
        # 1. Create the subplot figure
        # We create the figure first, then plot directly onto each axis
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        axes = axes.flatten()
        
        # 2. Package atoms objects and their titles
        structures = [atoms_1, atoms_2, atoms_3, atoms_4, atoms_1, atoms_2, atoms_3, atoms_4]
        titles = [
            "1. Initial Slab",
            "2. Relaxed Slab",
            "3. Adsorbate Added",
            "4. Final Relaxed System",
            "1. Initial (Top View)",
            "2. Relaxed (Top View)",
            "3. Adsorbate (Top View)",
            "4. Final (Top View)"
        ]
        
        # 3. Define common plotting arguments
        plot_kwargs_side = {
            'rotation': '-90x',      # Side view
        }

        plot_kwargs_top = {
            'rotation': '0x,0y,0z'      # Top-down view
        }

        # 4. Loop and plot directly onto each subplot axis
        for i, (ax, atoms, title) in enumerate(zip(axes, structures, titles)):
            # Use ase.visualize.plot.plot_atoms to draw on the axis
            if i < 4: # First row is side view
                plot_atoms(
                    atoms,
                    ax,
                    **plot_kwargs_side
                )
            else: # Second row is top view
                plot_atoms(
                        atoms,
                        ax,
                        **plot_kwargs_top
                    )
            ax.set_title(title, fontsize=12)
            ax.set_axis_off() # Turn off the black box border

        # 5. Set overall figure properties and save
        fig.suptitle("Simulation Geometry Progression", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, dpi=300) # Save with high resolution
        
        print(f"[Tool] Successfully saved comparison plot to {filename}")
        
    except Exception as e:
        print(f"[Tool] Error plotting structure comparison: {e}")
        # Print full error for debugging
        traceback.print_exc()
        return f"Error: {e}"
    
    return f"Plot saved to {filename}"
