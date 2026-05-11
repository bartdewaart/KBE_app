import os

import pandas as pd
from parapy.gui import display

from src.propulsion_system import PropulsionSystem


def load_inputs(file_path):
    """
    Integration Rule: reads mission specifications from Excel file.
    Returns a pandas Series indexed by Parameter column,
    accessible as a dict via specs['MTOW'] etc.
    Expected columns: Parameter, Value
    Expected rows: MTOW, n_rotors, safety_margin, max_diameter
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Mission input file not found at '{file_path}'. "
            f"Expected location: data/input/mission.xlsx. "
            f"Check that the file exists relative to the project root."
        )
    return pd.read_excel(file_path).set_index('Parameter')['Value']


if __name__ == '__main__':

    # Step 1: load mission specifications from Excel
    inputs = load_inputs("./data/input/mission.xlsx")

    # Extend inputs to include battery specs and weights
    battery_specs = {
        'energy_density': inputs.get('Battery Energy Density', 200),
        'capacity': inputs.get('Battery Capacity', 1000)
    }
    weights = {
        'endurance': inputs.get('Weight Endurance', 1),
        'efficiency': inputs.get('Weight Efficiency', 1),
        'weight': inputs.get('Weight Weight', 1)
    }
    inputs['battery_specs'] = battery_specs
    inputs['weights'] = weights

    # Step 2: instantiate the root PropulsionSystem object
    app = PropulsionSystem(specs=inputs)

    # Step 3: run optimization — finds optimal airfoil, blade count,
    # diameter and RPM, then applies best values back to the model
    #app.global_optimization

    # Step 4: print design report summary
    app.generate_report

    # Step 5: launch ParaPy GUI — geometry reflects optimal design
    display(app)