# Report of Agentic Pipeline for IR Spectra of Water on Metal Surfaces
## Outline:
 - A. Agent architecture and workflow
 - B. Method choice & rationale (e.g., DFT vs DFTB vs MLIP)
 - C. IR spectrum plot and explanation (with axes labeled)
 - D. Uncertainty estimation on at least one spectral feature (e.g., ensemble or sensitivity test)

## A. Agent architecture and workflow

The Agent architecture are shown as below:

![langgraph_flowchart](langgraph_structures_v3.png)

The details of Agent architecture are include in `README.md`

Please check section 3 in `README.md` "3. HOW IT WORKS: AGENT ARCHITECTURE" 

For the detailed usage workflow, please also take look at "5. USAGE"

## B. Method choice & rationale (e.g., DFT vs DFTB vs MLIP)

Both DFT and DFTB are connected to this graphs, while MILP are not included due to the time limitation. While UMA from META / MACE should be included as comparsion. Please check the "2. CORE FEATURES" and "Calculator Performance Summary: Single-Core Benchmark." from "5. USAGE" for further details.

The calculation of IR spectra for molecules adsorbed on surfaces presents a significant computational challenge, requiring a delicate balance between accuracy and efficiency. The choice of calculator is paramount, as vibrational frequencies are derived from the Hessian matrix (second derivatives of energy), making them highly sensitive to the quality of the underlying potential energy surface and the description of bonding between the adsorbate and the surface.

My workflow implements a _get_calculator function to provide a tiered selection of methods, allowing the user to match the computational cost to the scientific objective:

High-Accuracy (DFT): For final, publishable-quality spectra, Density Functional Theory (DFT) is the standard. It provides a robust quantum mechanical description of the electronic structure, charge transfer, and covalent/ionic interactions essential for accurate vibrational modes. In this implementation, this tier is represented by VASP (set as the 'default'/'accurate' option), GPAW, and NWChem. The primary drawback is their high computational cost, which can be prohibitive for large surface models or high-throughput screening.

Fast, Approximate (Semi-Empirical Tight-Binding): For rapid exploration, initial structure pre-optimization, or qualitative trend analysis, semi-empirical methods offer a compromise. This tier includes GFN2-xTB (via 'xtb'), GFN1-xTB (via 'tblite'), and DFTB+. These methods (which occupy a similar niche to DFTB or fast MLIPs) are orders of magnitude faster than DFT. However, their accuracy for subtle surface-adsorbate interactions (like dispersion or d-band coupling) may be limited, requiring careful validation against DFT benchmarks for the specific system.

Workflow Testing (Mock Calculator): The EMT ('fast') calculator is a "mock" calculator. It is not intended for generating physically meaningful results. Its role is strictly for debugging and testing the computational workflow itself (e.g., to check if optimization loops or file-parsing scripts run without error) without incurring the time cost of an actual simulation.

The strategic choice of method is therefore crucial. A typical high-fidelity workflow might use a fast method like xtb for an initial structural relaxation before switching to VASP or GPAW for the final, high-accuracy frequency calculation.

Due to project requirement set the limitaion of computaional resource merely on local laptop. 
Therefore, most of the result I presents are based on merely based on DFTB+ or EMT, so it is difficult to compared the for this specific case.


| Feature | **DFT (Density Functional Theory)** | **DFTB (Density-Functional Tight-Binding)** | **MLIP (Machine Learning Interatomic Potential)** |
| :--- | :--- | :--- | :--- |
| **Underlying Principle** | Solves approximate forms of the quantum mechanical electronic Schrödinger equation. | An approximate, semi-empirical version of DFT based on a minimal atomic orbital basis set and pre-calculated parameters. | An empirical potential (like classical force fields) where the interaction function is a flexible machine-learning model (e.g., neural network) trained on *ab initio* data. |
| **Example Calculators** | VASP, GPAW, NWChem, Quantum ESPRESSO | DFTB+, GFN1-xTB (via `tblite`), GFN2-xTB (via `xtb`) | Nequip, MACE, DeepMD, ANI (Note: Requires a pre-trained model for the specific chemical system). |
| **Typical Accuracy** | **High.** Considered the "gold standard" for chemical accuracy in solid-state and surface science (with appropriate functional). | **Medium to Low.** Good for structures and qualitative trends. Less reliable for energies, charge transfer, and non-covalent interactions unless parameterized for them. | **High to Very High.** Can *reproduce the accuracy of the training data* (e.g., DFT). Accuracy is entirely dependent on the quality and completeness of the training set. |
| **Computational Cost** | **Very High.** Scales poorly (e.g., $O(N^3)$) with system size. Vibrational calculations are extremely demanding. | **Fast.** Scales much better with system size. Significantly (100-1000x+) faster than DFT. | **Very Fast.** Speed is comparable to or faster than DFTB, often approaching classical potentials. Offers *DFT-level accuracy at near-classical speed*. |
| **Pros** | • Highly accurate & predictive.<br>• General-purpose; no system-specific training needed (beyond choosing a functional).<br>• Correctly describes quantum effects like bond breaking/forming and charge transfer. | • Excellent balance of speed and *reasonable* quantum mechanics.<br>• Good for pre-optimization, high-throughput screening, and large systems (1000s of atoms). | • *Potential* for DFT or higher accuracy.<br>• Extremely fast, enabling large-scale molecular dynamics (MD) or massive system sizes for frequency calculations. |
| **Cons / Limitations** | • Extremely slow and computationally expensive.<br>• Limited to smaller systems (typically <1000 atoms) and short timescales. | • Accuracy is limited by parameterization. May fail for systems/interactions not in its parameter set.<br>• Less reliable for vibrational frequencies than DFT. | • **Requires a large, high-quality, system-specific training dataset** (e.g., from thousands of DFT calculations).<br>• Creating this training set is a massive, time-consuming effort in itself.<br>• Not "predictive" for new chemical environments it wasn't trained on. |
| **Use Case for Surface IR**| • **Final Accuracy:** Calculating the final, high-fidelity IR spectrum for publication.<br>• **Benchmark Data:** Generating the "ground truth" data needed to train an MLIP. | • **Pre-Optimization:** Finding a reasonable starting geometry for a full DFT calculation.<br>• **Qualitative Screening:** Quickly comparing trends across many different adsorbates or sites. | • **Large-Scale Dynamics:** Simulating temperature effects on the IR spectrum via MD.<br>• **Complex Systems:** Calculating spectra for very large surface models or nanoparticles where DFT is impossible.<br>• **Once Trained:** High-throughput prediction of spectra. |


## C. IR spectrum plot and explanation (with axes labeled)

For the water adsorption on Ag case, 
To start with, due to computaional resource limitation, the calculation is conducted with dftb+ based on paraemters set 

For this workflow, the DFTB+ calculator is configured with two distinct parameter sets depending on the task: geometry optimization ('opt') or vibrational frequency calculation ('freq'). Both modes employ the same foundational Hamiltonian settings, including a maximum angular momentum (basis set) of 's' for Hydrogen, 'p' for Carbon, Nitrogen, and Oxygen, and 'd' for Sulfur, Phosphorus, and Gold. Both also utilize a Gamma-point-only k-point mesh (kpts=(1, 1, 1)) for computational speed (suitable for large supercells or non-periodic systems) and a maximum of 500 Self-Consistent-Charge (SCC) iterations.

The critical distinction between the modes lies in the geometry convergence criteria. For the 'opt' mode, a standard Conjugate Gradient driver is used with a maximum force component tolerance of 1e-4 (in ASE's default units). For the 'freq' mode, this tolerance is tightened significantly to 1e-6. (should be 1e-8 if possible) This much stricter convergence is necessary because calculating vibrational frequencies requires a high-precision Hessian matrix (second derivatives of energy), which is numerically stable and accurate only when the structure is converged to a precise local minimum.

Due to lack of IR active/Inactive in the DFTB calculator, I consider all the modes to be IR active for simplifcation.  (which present in VASP and GPAW calculators)


1. Geometry Optimization
First, the script performs a geometry optimization on the provided atomic structure (atoms). It initializes the L-BFGS optimization algorithm, which is designed to find a stable, low-energy configuration for the atoms. The optimization process is set to run until the maximum force (fmax) on any atom in the system falls below a predefined threshold (optimizer_fmax). All output and progress from this optimization step are saved to a log file.

2. Vibrational Analysis Preparation
After the optimization is complete, the script prepares for a vibrational analysis. To reduce computational cost, the analysis is limited to a specific subset of atoms (e.g., an adsorbate molecule like water) rather than the entire system. It identifies the indices of these atoms, assuming they are the last atoms added to the structure.

Before starting the new calculation, the script archives any old results. It checks if a default output directory named vib exists. If it does, the directory is renamed with a unique timestamp (e.g., vib_archive_20251104_195527) to prevent the new calculation from overwriting the old data.

3. Conditional Vibrational Calculation
The script then uses a conditional check to determine how to run the vibrational analysis, based on the simulation calculator being used:

Path A (VASP or GPAW): If the calculator is VASP or GPAW, the script assumes the calculator can compute dipole moments, which are necessary for true infrared (IR) intensities. It initializes and runs a full Infrared calculation using the selected atomic indices and a specified finite displacement (vib_delta). After the calculation, it prints a summary of the IR modes and returns the final Infrared object.

Path B (Other Calculators): If the calculator is not VASP or GPAW (e.g., EMT or DFTB+, which lack dipole moment support), the script defaults to a standard Vibrations calculation. This calculation is run on the same atomic indices and also prints a summary. However, this method only provides vibrational frequencies, not their IR intensities.

4. Spectrum Plotting and Broadening (for Path B)
If the standard Vibrations analysis (Path B) was performed, the script post-processes the results to create a plot.

Extract Frequencies: It retrieves the calculated frequencies, takes their real part (as ASE provides complex numbers), and converts the units from meV to wavenumbers (cm⁻¹). It also filters out any low-frequency "noise" modes below 50 cm⁻¹.

Mock Intensities: Since this path does not have official IR intensities, the script creates mock intensities by assigning a value of 1 to every valid frequency.

Gaussian Broadening: To simulate the appearance of a real-world experimental spectrum, each "stick" frequency is broadened using a Gaussian function. The script generates a final spectrum by summing these individual Gaussian curves.

Plotting: This final, broadened spectrum is then plotted on a pre-existing matplotlib axis (ax) with a label for identification.

- #### H2O_on_Au_fcc_100_dftb+
- Example of dftb+ [system_name]_structures.png :
![Example of pure molecule](results/H2O_on_Au_fcc_100_dftb+_structures.png)

- Example of dftb+ [system_name]_ir.png :
![Example of pure molecule](results/H2O_on_Au_fcc_100_by_dftb+_ir.png)


