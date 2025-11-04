import os
from langgraph.graph import StateGraph, END

# Import the state and node functions from planner.py
from planner import (
    SimulationState,
    planner_node,
    geometry_node,
    calculation_node,
    plot_geometry_node,
    spectra_node,
    analysis_node
)

# ---  Define the Graph ---
# This is where we wire the nodes together.

print("Building graph...")
workflow = StateGraph(SimulationState)

# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("geometry", geometry_node)
workflow.add_node("calculation", calculation_node)
workflow.add_node("plot_geometry", plot_geometry_node)
workflow.add_node("spectra", spectra_node)
workflow.add_node("analysis", analysis_node)

# Add edges to define the workflow
workflow.set_entry_point("planner")
workflow.add_edge("planner", "geometry")
workflow.add_edge("geometry", "calculation")
workflow.add_edge("calculation", "plot_geometry") # Plot the structures
workflow.add_edge("plot_geometry", "spectra")     # Then generate spectrum
workflow.add_edge("spectra", "analysis")
workflow.add_edge("analysis", END)

# Compile the graph into a runnable application
# This 'app' object will be imported by test_endtoend.py
app = workflow.compile()

print("Graph compiled successfully.")


### --- Optional: Visualize the Graph ---
# This part will run when this file is imported, saving the graph image.
try:
    png_bytes = app.get_graph().draw_mermaid_png()
    output_file = "./langgraph_structures_v3.png"
    with open(output_file, "wb") as f:
        f.write(png_bytes)
    print(f"\n Graph visualization saved to: {os.path.abspath(output_file)}")
except Exception as e:
    print(f"\nCould not generate graph visualization: {e}")
    print("Please ensure 'mermaid-cli' or 'pyppeteer' is installed if you want a visual.")
