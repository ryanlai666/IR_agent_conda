import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import TypedDict, Optional
import getpass

# Langchain & LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# ASE imports
from ase.atoms import Atoms
from ase.io import read, write
from ase.vibrations import Vibrations

# Import custom scientific tools
import simulation_tools as sim_tools
import postprocess_tools as post_tools


# --- Setup Tools & LLM ---
# This logic is moved here from build_graph.py
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("GOOGLE_API_KEY not found in .env file.")
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Define the LLM for the analysis agent
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Use fast model for structured output
    temperature=0,
    convert_system_message_to_human=True # Helper for system prompts
)



# --- Define Agent State ---

class SimulationState(TypedDict):
    """
    Represents the state of our simulation workflow.
    """
    task: str                 # The initial user prompt
    
    calculator_level: str     # e.g., 'fast', 'default', 'accurate'
    surface_metal: str        # e.g., 'Au', 'Pt'
    adsorbate_formula: str    # e.g., 'H2O', 'CO'
    surface_metal_structure_type: str # e.g., 'fcc', 'bcc', 'hcp'
    surface_metal_facet: str  # e.g., '111', '100'
    
    # Geometry state
    system_str: str
    initial_slab: Optional[Atoms]
    relaxed_slab: Optional[Atoms]
    adsorbed_slab: Optional[Atoms]
    final_system: Optional[Atoms] # System with adsorbate, before final opt
    
    # Calculation state
    atoms: Optional[Atoms]    # The final optimized ASE atoms object
    vib_run: Optional[Vibrations] # The vibration object from the calculation
    
    # Spectra state
    plot_filename: str
    spectrum_data: dict
    
    # Analysis state
    analysis_report: str


class PlannerOutput(BaseModel):
    """Structured output for the Planner node."""
    calculator_level: str = Field(description="The computational cost/accuracy. One of: 'fast' (EMT), 'xtb', 'tblite', 'paw', 'nwchem', 'default' (VASP-medium), or 'accurate' (VASP-high).")
    surface_metal: str = Field(description="The elemental symbol for the metal surface, e.g., 'Au', 'Pt', 'Cu'.")
    adsorbate_formula: str = Field(description="The chemical formula of the adsorbate molecule, e.g., 'H2O', 'CO', 'NH3'.")
    surface_metal_structure_type: str = Field(description="The crystal structure type of the metal surface, e.g., 'fcc', 'bcc', 'hcp'.")
    surface_metal_facet: str = Field(description="The crystallographic facet of the metal surface, e.g., '111', '100', '110'.")

# Bind the Pydantic model to the LLM for structured output
# This is used by planner_node
structured_llm = llm.with_structured_output(PlannerOutput)

# --- Node Functions ---

def planner_node(state: SimulationState):
    """
    The "Planner" agent, powered by an LLM.
    Parses the user's task to determine simulation parameters.
    """
    print("--- (1) PLANNER NODE ---")
    task = state['task']
    print(f"Task: {task}")

    # Define the prompt for the planner
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert computational chemistry assistant. "
         "Your job is to parse a user's request and extract the key simulation parameters. "
         "You must identify the calculator level, the surface metal, the surface most likely crystal structure type, the surface metal facet, and the adsorbate molecule. "
         
         "Calculator levels have specific keywords:\n" 
         "- 'fast', 'cheap', 'emt': map to 'fast' (EMT)\n"
         "- 'xtb', 'gfn2': map to 'xtb'\n"
         "- 'tblite', 'gfn1': map to 'tblite'\n"
         "- 'dftb+', 'DFTB+': map to 'dftb+'\n"
         "- 'nwchem', 'NWChem','pbe': map to 'nwchem'\n"
         "- 'gpaw', 'paw': map to 'gpaw'\n"
         "- 'default', 'medium', 'dft', 'vasp': map to 'default' (VASP-medium)\n"
         "- 'accurate', 'high', 'precise', 'vasp-high': map to 'accurate' (VASP-high)\n"
         "If the user doesn't specify, use 'fast' for the calculator, 'Au' for the metal, and 'H2O' for the molecule as defaults.\n"
         "If the user specify not slab or Not adsorption, use 'None' for the metal as defaults."),
        ("human", "User task: {task}")
    ])
    
    # Create the planner chain
    # 'structured_llm' is now defined at the top of this file
    planner_chain = planner_prompt | structured_llm
    
    print("Parsing task with LLM ...")
    try:
        plan = planner_chain.invoke({"task": task})
        
        print(f" > Calculator: {plan.calculator_level}")
        print(f" > Metal: {plan.surface_metal}")
        print(f" > surface_metal_structure_type: {plan.surface_metal_structure_type}")
        print(f" > surface_metal_facet: {plan.surface_metal_facet}")
        print(f" > Adsorbate: {plan.adsorbate_formula}")

        print("Plan: geometry -> calculation -> plot_geometry -> spectra -> analysis")
        
        return {
            "plot_filename": f"results/{plan.adsorbate_formula}_on_{plan.surface_metal}_{plan.surface_metal_structure_type}_{plan.surface_metal_facet}_by_{plan.calculator_level}_ir.png",
            "analysis_report": f"results/{plan.adsorbate_formula}_on_{plan.surface_metal}_{plan.surface_metal_structure_type}_{plan.surface_metal_facet}_by_{plan.calculator_level}_report.txt",
            "spectrum_data": {},
            "calculator_level": plan.calculator_level,
            "surface_metal": plan.surface_metal,
            "surface_metal_structure_type": plan.surface_metal_structure_type,
            "surface_metal_facet": plan.surface_metal_facet,
            "adsorbate_formula": plan.adsorbate_formula,
            "system_str":"system_SSSSS"
        }
    except Exception as e:
        print(f"Error in planner node: {e}")
        return {
            "analysis_report": f"Failed to plan task: {e}"
        }

def geometry_node(state: SimulationState):
    """
    The "Geometry" agent.
    - Sets up the initial slab.
    - Optimizes the bare slab.
    - Adds the adsorbate and constraints.
    """
    print("--- (2) GEOMETRY NODE ---")
    
    metal = state['surface_metal']
    surface_metal_facet = state['surface_metal_facet']
    surface_metal_structure_type = state['surface_metal_structure_type']
    adsorbate = state['adsorbate_formula']
    calculator_preset = state['calculator_level']
    
    ## pure mol route
    if metal=='none' or  metal=='None' or metal==None:
        system_str_geo_node= adsorbate+'_'+calculator_preset 
        print(f"Setting up pure molecule ...")
        
        init_mol = sim_tools.setup_molecule_tool.func( 
            adsorbate_formula=adsorbate)
        
        print(f"Geometry setup complete. Total atoms: {len(init_mol)}")        
        
        print(f"Optimizing molecule with '{calculator_preset}' calculator...")
        relaxed_mol = sim_tools.run_opt_tool.func(
            init_mol.copy(),
            calculator_preset=calculator_preset,
            system_name=system_str_geo_node+'_molecule'
        )
        
        xyz_filename= system_str_geo_node+'_molecule.xyz'
        write(xyz_filename, relaxed_mol)
        print(f"Geometry saved to xyz:",xyz_filename)
        
        return {
            "atoms": relaxed_mol, 
            "initial_slab": init_mol,
            "relaxed_slab": init_mol,
            "adsorbed_slab": init_mol, 
            "final_system": relaxed_mol,
            "system_str": system_str_geo_node
        }
    
    ## slab route
    else:
        system_str_geo_node= adsorbate+'_on_'+metal+'_'+surface_metal_structure_type+'_'+surface_metal_facet+'_'+calculator_preset
    
        print(f"Setting up {metal} slab...")
        initial_slab = sim_tools.setup_slab_tool.func(surface_metal=metal, 
                                surface_metal_structure_type=surface_metal_structure_type,
                                surface_metal_facet=surface_metal_facet, slab_size=(3, 3, 3), preset=calculator_preset)
        
        print(f"Optimizing bare {metal} slab with '{calculator_preset}' calculator...")
        relaxed_slab = sim_tools.run_opt_tool.func(
            initial_slab.copy(),
            calculator_preset=calculator_preset,
            system_name=system_str_geo_node+'_bare_slab'
        )
        print("Bare slab optimization complete.")

        print(f"Adding {adsorbate}...")
        ads_system = sim_tools.setup_adsorbate_tool.func(
            slab=relaxed_slab.copy(), 
            adsorbate_formula=adsorbate
        )

        final_system = sim_tools.run_opt_tool.func(
            ads_system.copy(),
            calculator_preset=calculator_preset,
            system_name=system_str_geo_node+'_slab_ads'
        )
        
        print(f"Geometry setup complete. Total atoms: {len(final_system)}")        
        
        xyz_filename= system_str_geo_node+'_slab.xyz'
        write(xyz_filename, final_system)
        print(f"Geometry saved to xyz:",xyz_filename)

        return {
            "atoms": final_system,
            "initial_slab": initial_slab,
            "relaxed_slab": relaxed_slab,
            "adsorbed_slab": ads_system,
            "final_system": final_system,
            "system_str": system_str_geo_node
        }

def calculation_node(state: SimulationState):
    """
    The "Calculation" agent.
    - Runs the final geometry optimization on the combined system.
    - Runs the frequency (vibrational) analysis.
    """
    print("--- (3) CALCULATION NODE ---")
    atoms = state["atoms"].copy()
    adsorbate_formula = state['adsorbate_formula']
    calculator_preset = state['calculator_level']
    
    print(f"Running final optimization and frequency analysis with '{calculator_preset}' calculator...")
    opt_atoms, vib_run = sim_tools.run_opt_freq_tool.func(
        atoms=atoms.copy(),
        calculator_preset=calculator_preset,
        optimizer_fmax=0.01,
        vib_delta=0.01,
        adsorbate_formula=adsorbate_formula,
        system_name=state["system_str"]+'_opt_freq'
    )
    
    print("Optimization and frequency calculation complete.")
    
    return {"atoms": opt_atoms , "vib_run": vib_run}

def plot_geometry_node(state: SimulationState):
    """
    A utility agent to plot the geometry comparison.
    """
    print("--- (4) PLOT GEOMETRY NODE ---")
    try:
        plot_filename = f"results/"+state["system_str"]+"_structures.png"
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        
        post_tools.plot_structure_comparison_tool.func(
            atoms_1=state["initial_slab"],
            atoms_2=state["relaxed_slab"],
            atoms_3=state["adsorbed_slab"],
            atoms_4=state["final_system"],
            filename=plot_filename
        )
        print(f"Structure comparison plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error plotting geometry: {e}")
    
    return {}

def spectra_node(state: SimulationState):
    """
    The "Spectra" agent.
    - Extracts frequency data from the calculation.
    - Generates and saves the IR spectrum plot.
    """
    print("--- (5) SPECTRA NODE ---")
    vib_run = state["vib_run"]
    
    metal = state['surface_metal']
    adsorbate = state['adsorbate_formula']
    calc_level = state['calculator_level']

    fig, ax = plt.subplots(figsize=(10, 6))

    spectrum_data = post_tools.extract_and_plot_spectra_tool.func(
        vib_run=vib_run,
        run_label=f"{calc_level} (delta=0.01)",
        color="#7645B3",
        ax=ax
    )

    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax.set_ylabel("Mock IR Intensity (a.u.)", fontsize=12)

    ## for pure moleclue case
    if metal==None or metal=="None" or metal=="none":
        ax.set_title(f"IR Spectrum of {adsorbate} molecule", fontsize=14)
        plot_filename=f"results/{adsorbate}_{calc_level}_mol_ir.png"
    
    # for adsorption case
    else:
        ax.set_title(f"IR Spectrum of {adsorbate} on {metal}(111) (LangGraph Run)", fontsize=14)
        plot_filename = state["plot_filename"] 
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename)
    print(f"Spectrum plot saved to: {plot_filename}")

    return {"spectrum_data": spectrum_data}

def analysis_node(state: SimulationState):
    """
    The "Analysis" agent.
    - Takes the final numerical data.
    - Calls an LLM to generate a human-readable report.
    """
    print("--- (6) ANALYSIS NODE ---")
    spectrum_data = state["spectrum_data"]
    
    metal = state['surface_metal']
    adsorbate = state['adsorbate_formula']
    calc_level = state['calculator_level']
    
    report_filename = state['analysis_report']  ## saved report filename

    freqs_cm = spectrum_data.get("frequencies_cm", [])
    freqs_str = ", ".join([f"{f:.1f}" for f in freqs_cm])
    data_summary = (
        f"Simulation Parameters:\n"
        f"- System: {adsorbate} on {metal}\n"
        f"- Calculator: {calc_level}\n\n"
        f"Peak Frequencies (cm⁻¹) from single run: [{freqs_str}]\n"
    )

    analysis_prompt = ChatPromptTemplate.from_template(
        "You are a computational chemistry assistant. "
        "A user has run a *mock* simulation for an IR spectrum. "
        "The calculator used is for demonstration, so the peak positions may not be physically accurate "
        "(especially EMT, XTB, and TBLite, which lack dipole info). "
        "Your job is to analyze the *results* of the simulation, not their physical accuracy. "
        "Please provide a brief, 1-paragraph summary of the findings based on the data provided.\n\n"
        "Data:\n{data}\n\n"
        "Analysis:"
    )
    
    # 'llm' is now defined at the top of this file
    analysis_chain = analysis_prompt | llm | StrOutputParser()
    
    print("Calling LLM for final analysis...")
    analysis_response = analysis_chain.invoke({"data": data_summary})
    
    print("\n[LangChain Agent Analysis]:")
    print(analysis_response)

    # --- ADDED FILE SAVING LOGIC ---
    try:
        # Ensure the 'results' directory exists
        os.makedirs(os.path.dirname(report_filename), exist_ok=True)
        
        # Save the analysis_response content to the file
        with open(report_filename, 'w') as f:
            f.write(analysis_response)
        
        print(f"Analysis report saved to: {report_filename}")
    
    except Exception as e:
        print(f"Error saving analysis report to {report_filename}: {e}")
    # --- END OF ADDED LOGIC ---


    return {"analysis_report": analysis_response}
