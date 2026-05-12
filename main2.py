import os
import pandas as pd
from parapy.gui import display
from src.propulsion_system import PropulsionSystem

def load_mission_specs(file_path):
    """
    Integration Rule: Reads external design specs.
    Converts mission constraints from Excel/CSV into a dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at '{file_path}'.")

    # Determine file type and read data
    _, ext = os.path.splitext(file_path.lower())
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Transform: Set 'Parameter' as index, extract 'Value', and convert to dict
    return df.set_index('Parameter')['Value'].to_dict()

if __name__ == '__main__':
    # 1. Load mission specs from external file
    mission_path = "./data/input/mission.xlsx"
    if not os.path.exists(mission_path):
        mission_path = "./data/input/mission.csv"
    
    specs_dict = load_mission_specs(mission_path)

    # 2. Instantiate the system with mission specs
    obj = PropulsionSystem(specs=specs_dict)
    
    # 3. Run the normalized optimization 
    print("Starting Optimization...")
    obj.optimize_design()

    # 4. Print final analysis results [cite: 93, 106]
    perf = obj.propeller.performance
    print(f"Final Thrust: {perf['thrust']:.2f} N")
    print(f"Power Required: {perf['power']:.2f} W")

    # 5. Visualize the 3D Lofted Geometry [cite: 70, 106]
    display(obj)