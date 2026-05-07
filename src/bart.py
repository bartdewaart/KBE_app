import pandas as pd
import numpy as np
import math
import os
import subprocess
from parapy.core import Base, Input, Attribute, Part, Sequence
from scipy.interpolate import interp1d
from scipy.optimize import minimize


# --- 1. DATA INGESTION ---
def load_uav_specs(file_path):
    """Integration Rule: Reads external design specs[cite: 87]."""
    mission = pd.read_excel(file_path, sheet_name="Mission_Constraints").set_index('Parameter')['Value']
    motors = pd.read_excel(file_path, sheet_name="Motor_Database")
    return mission, motors


# --- 2. AERODYNAMICS ---
class Airfoil(Base):
    naca_code = Input("4412")
    reynolds = Input(300000)

    @Attribute
    def polar_data(self):
        """Runs XFOIL subprocess and identifies alpha_opt[cite: 88, 110]."""
        # (XFOIL execution and parsing logic same as previous versions)
        # Returns {cl_interp, cd_interp, alpha_opt, cl_opt}
        pass


# --- 3. GENERATIVE BLADE SECTION ---
class BladeSection(Base):
    radius = Input()
    dr = Input()
    total_radius = Input()
    n_blades = Input()
    rpm = Input()
    airfoil_obj = Input()
    target_thrust = Input()

    @Attribute
    def optimum_geometry(self):
        """Generates geometry based on Betz Optimality [cite: 26, 74-78]."""
        disk_area = math.pi * self.total_radius ** 2[cite: 29]
        vi = math.sqrt(self.target_thrust / (2.0 * 1.225 * disk_area))[cite: 28]

        omega = self.rpm * 2 * math.pi / 60
        v_eff = math.sqrt(vi ** 2 + (omega * self.radius) ** 2)[cite: 33]
        phi = math.atan2(vi, omega * self.radius)[cite: 34]

        # Optimum Pitch Rule [cite: 75]
        pitch = phi + self.airfoil_obj.polar_data["alpha_opt"]

        # Optimum Chord Rule [cite: 78]
        f_tip = (self.n_blades / 2) * (self.total_radius - self.radius) / (self.radius * math.sin(phi))[cite: 56]
        F = (2 / math.pi) * math.acos(math.exp(-f_tip))[cite: 57]

        cl_opt = self.airfoil_obj.polar_data["cl_opt"]
        chord = (8 * math.pi * self.radius * vi ** 2 * F) / (self.n_blades * v_eff ** 2 * cl_opt * math.cos(phi))

        return {"chord": max(0.01, chord), "pitch": pitch}

    @Attribute
    def forces(self):
        """Calculates dT and dQ using BEMT [cite: 47-48]."""
        # (Returns sectional thrust and torque)
        pass


# --- 4. THE PROPELLER ASSEMBLY ---
class HoverPropeller(Base):
    diameter = Input()
    rpm = Input()
    target_thrust = Input()
    n_segments = Input(15)

    @Part
    def airfoil(self):
        return Airfoil(naca_code=self.airfoil_type)

    @Attribute
    def smooth_distributions(self):
        """Generates smooth curves through optimal points."""
        # 1. Define 5 control stations [cite: 71, 72]
        r_stations = np.linspace(self.hub_diameter / 2, self.diameter / 2, 5)

        # 2. Calculate 'Raw' optimal chord and pitch at these 5 points [cite: 74-78]
        raw_chords = [self.calc_opt_chord(r) for r in r_stations]
        raw_pitches = [self.calc_opt_pitch(r) for r in r_stations]

        # 3. Fit Cubic Splines
        chord_spline = CubicSpline(r_stations, raw_chords)
        pitch_spline = CubicSpline(r_stations, raw_pitches)

        return chord_spline, pitch_spline

    @Part(parse=False)
    def sections(self):
        """Discretization Rule: Instantiates blade segment objects[cite: 83]."""
        return Sequence(type=BladeSection, quantify=self.n_segments,
                        target_thrust=self.target_thrust, rpm=self.rpm,
                        radius=lambda i: 0.02 + (i.index + 0.5) * ((self.diameter / 2 - 0.02) / self.n_segments),
                        airfoil_obj=self.airfoil, total_radius=self.diameter / 2,
                        dr=(self.diameter / 2 - 0.02) / self.n_segments, n_blades=2)

    @Attribute
    def power_required(self):
        torque = sum(s.forces["dQ"] for s in self.sections)
        return torque * (self.rpm * 2 * math.pi / 60)[cite: 574]


# --- 5. THE MASTER SYSTEM ---
class PropulsionSystem(Base):
    specs = Input()  # Mission constraints from Excel
    motors = Input()  # Motor DB from Excel

    @Attribute
    def thrust_required(self):
        """Design Thrust Rule: Computes target per rotor [cite: 22-23]."""
        return (self.specs['MTOW'] * 9.81 / self.specs['n_rotors']) * self.specs['safety_margin']

    @Attribute
    def optimal_design(self):
        """Outer Loop: Propeller Optimization via Scipy ."""

        def objective(x):
            self.propeller.diameter = x[0]
            self.propeller.rpm = x[1]
            return self.propeller.power_required

        # Minimizing power by varying diameter and RPM
        res = minimize(objective, [0.3, 5000], bounds=[(0.1, self.specs['max_diameter']), (2000, 8000)])
        return {"diameter": res.x[0], "rpm": res.x[1], "power": res.fun}

    @Part
    def propeller(self):
        return HoverPropeller(target_thrust=self.thrust_required)

    @Attribute
    def motor_matching(self):
        """Inner Loop: Coupled Feasibility Evaluation [cite: 586-600]."""
        feasible_motors = []
        p_req = self.propeller.power_required
        q_req = p_req / (self.propeller.rpm * 2 * math.pi / 60)[cite: 593]

        for _, m in self.motors.iterrows():
            v_req = self.propeller.rpm / m['KV'][cite: 591]
            i_req = q_req / (1 / (m['KV'] * 2 * math.pi / 60))[cite: 595]

            # Feasibility Rules [cite: 596-599]
            if v_req < 25 and i_req < m['Max_Power'] / v_req:
                feasible_motors.append(m['Model'])
        return feasible_motors

    def generate_pdf_report(self):
        """Analysis Rationale: Documents design reasoning[cite: 105, 794]."""
        print(f"REPORT GENERATED\nTarget Thrust: {self.thrust_required:.2f} N")
        print(f"Optimal Prop: D={self.propeller.diameter:.2f}m @ {self.propeller.rpm} RPM")
        print(f"Feasible Motors: {self.motor_matching}")


if __name__ == "__main__":
    mission_data, motor_db = load_uav_specs("input_data.xlsx")
    app = PropulsionSystem(specs=mission_data, motors=motor_db)
    app.generate_pdf_report()