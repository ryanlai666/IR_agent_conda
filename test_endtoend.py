# This is the main file you run: python test_endtoend.py

# Import the compiled graph 'app' from build_graph.py
from build_graph import app

if __name__ == "__main__":
    
    #  --- CHOOSE/CHANGE YOUR TASK HERE ---
    
    #initial_task = "Run a IR spectrum simulation for H2O on Au(111) and analyze the results."  
    
    # 
    Take_task_from_user =True
    
    
    if Take_task_from_user:
        initial_task = input("\nEnter your intend IR simulation. *Notice: Currently the molecule will be on attached on top site.\n\n")
    else:
        # sucessfull case:
        #initial_task = "Run a IR spectrum simulation for NH3 on Au(100) and analyze the results."
        initial_task = "Run a IR spectrum simulation for N2 on Au(111) and analyze the results with fast caluclation."
        #initial_task = "Run a IR spectrum simulation for N2 on Au(111) and analyze the results with dftb+."
        #initial_task = "Run a IR spectrum simulation for H2O not metal and analyze the results with xtb." # pure molecule case
        
        #initial_task = "Run a IR spectrum simulation for N2 on Au(111) and analyze the results with GPAW."  ## slow
        #initial_task = "Run a IR spectrum simulation for N2 on Au(111) and analyze the results with nwchem"  ## take very long
        
        #initial_task = "I need to see the IR for CO on Platinum using dftb+."
    
    print('='*20+' human messenge '+'='*20+'\n')
    print(initial_task+'\n')
    print('='*20+' AI planner begins '+'='*20+'\n')
    

    print("\n--- Initializing LangGraph Multi-Agent Workflow ---")

    inputs = {"task": initial_task}
    
    # Run the graph
    # .stream() streams all state changes. We capture the final one.
    final_state = None
    for s in app.stream(inputs):
        final_state = s
        print("\n" + "="*40)

    print("\n--- LangGraph Workflow Complete ---")
    
    # Access the final state
    if final_state:
        final_report_key = list(final_state.keys())[0] # Get the key of the last node
        final_report_data = final_state[final_report_key]

        print("\nFinal Report:")
        print(final_report_data.get("analysis_report", "No report generated."))
        
        # Print plot filename from 'spectra' node, not 'analysis'
        if 'spectra' in final_state:
             print(f"\nFinal plot saved to: {final_state['spectra'].get('plot_filename', 'N/A')}")
        elif 'planner' in final_state:
             # Fallback to planner's predicted filename if spectra node fails
             print(f"\nPlot saved to: {final_state['planner'].get('plot_filename', 'N/A')}")

    else:
        print("Workflow did not complete successfully.")

