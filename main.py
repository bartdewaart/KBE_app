from parapy.gui import display
from src.propulsion_system import PropulsionSystem
import pandas as pd

def load_inputs(file_path):
    inputs_df = pd.read_excel(file_path).set_index('Parameter')['Value']
    return inputs_df

if __name__ == '__main__':
    inputs = load_inputs("./data/input/mission.xlsx")
    app = PropulsionSystem(specs=inputs)
    report = app.generate_report
    display(app)