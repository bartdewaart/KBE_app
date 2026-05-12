import os
import csv
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
            f"Expected location: data/input/mission.csv"
        )
    specs = {}
    with open(file_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        print(f"DEBUG mission headers: {reader.fieldnames}")
        for row in reader:
            # Convert numeric values to float, keep strings as-is
            try:
                specs[row["Parameter"]] = float(row["Value"])
            except ValueError:
                specs[row["Parameter"]] = row["Value"]
    return specs


if __name__ == '__main__':

    # Step 1: load mission specifications from Excel
    inputs = load_inputs("./data/input/mission.csv")

    # Step 2: instantiate the root PropulsionSystem object
    app = PropulsionSystem(specs=inputs)

    # Step 3: run optimization — finds optimal airfoil, blade count,
    # diameter and RPM, then applies best values back to the model
    app.run_optimization()

    # Step 4: print design report summary
    app.generate_report()

    # Step 5: launch ParaPy GUI — geometry reflects optimal design
    display(app)