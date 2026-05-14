import os
import csv

from parapy.gui import display
from src.propulsion_system import PropulsionSystem


def load_inputs(file_path):
    """
    Integration Rule: reads mission specifications from a CSV file.
    Returns a dict keyed by Parameter column.
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
            try:
                specs[row["Parameter"]] = float(row["Value"])
            except ValueError:
                specs[row["Parameter"]] = row["Value"]
    return specs


if __name__ == '__main__':

    # Step 1: load mission specifications from CSV
    inputs = load_inputs("./data/input/mission.csv")

    # Step 2: instantiate PropulsionSystem with each spec as a
    # first-class Input slot so the user can edit them in the GUI tree.
    app = PropulsionSystem(
        MTOW          = float(inputs["MTOW"]),
        n_rotors      = int(inputs["n_rotors"]),
        safety_margin = float(inputs["safety_margin"]),
        max_diameter  = float(inputs["max_diameter"]),
    )

    # Step 3: initial optimization. Inside the GUI the user can edit
    # any Input and right-click the PropulsionSystem node → "Re-run
    # optimization" to refresh the optimal design.
    app.run_optimization()
    app.generate_report()

    # Step 4: launch ParaPy GUI
    display(app)