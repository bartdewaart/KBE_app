import math
import subprocess
import os
import numpy as np
from parapy.core import Base, Input, Attribute, Part, Sequence
from scipy.interpolate import interp1d


# --- 1. AERODYNAMICS COMPONENT ---
class Airfoil(Base):
    """Handles XFOIL integration and aerodynamic data interpolation."""
    naca_code = Input("4412")
    reynolds = Input(300000)
    alpha_min = Input(-5.0)
    alpha_max = Input(20.0)
    alpha_step = Input(0.5)

    @Attribute
    def polar_interpolators(self):
        """Runs XFOIL via subprocess and parses the resulting polar file."""
        # Define the unique filename
        polar_file = f"polar_{self.naca_code}_{int(self.reynolds)}.txt"

        # Domain Protection: Remove old polar file if it exists [cite: 177-178]
        if os.path.exists(polar_file):
            os.remove(polar_file)
            print(f"Stale data removed: {polar_file}")

        # XFOIL command sequence
        commands = (
            f"NACA {self.naca_code}\n"
            "PANE\n"
            "OPER\n"
            f"VISC {self.reynolds}\n"
            "ITER 200\n"
            "PACC\n"
            f"{polar_file}\n\n"
            f"ASEQ {self.alpha_min} {self.alpha_max} {self.alpha_step}\n"
            "PACC\n"
            "QUIT\n"
        )

        # Execution logic (assuming XFOIL is in system PATH)
        process = subprocess.Popen(
            "xfoil", stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True
        )
        process.communicate(commands)

        raw_data = []
        if os.path.exists(polar_file):
            with open(polar_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            # Try converting the first column to float.
                            # If it's "alpha" or "Reynolds", this fails and skips the line.
                            alpha = float(parts[0])
                            cl = float(parts[1])
                            cd = float(parts[2])
                            raw_data.append((alpha, cl, cd))
                        except ValueError:
                            continue

        if not raw_data:
            raise RuntimeError(f"No data parsed from {polar_file}. Check XFOIL convergence.")

        # Logic Rule: Sort and remove duplicates (common in repeated XFOIL outputs)
        # Using a dictionary to keep only the last occurrence of each alpha
        unique_points = {item[0]: (item[1], item[2]) for item in sorted(raw_data)}

        alphas = sorted(unique_points.keys())
        cls = [unique_points[a][0] for a in alphas]
        cds = [unique_points[a][1] for a in alphas]

        # Scipy Optimizer Integration [cite: 90, 97]
        cl_interp = interp1d(alphas, cls, kind='linear', fill_value="extrapolate")
        cd_interp = interp1d(alphas, cds, kind='linear', fill_value="extrapolate")

        return {"cl": cl_interp, "cd": cd_interp}

    @Attribute
    def optimal_performance(self):
        """Identifies alpha_opt and cl_opt where L/D is maximum ."""
        data = self.polar_interpolators
        l_over_d = [cl / max(cd, 1e-6) for cl, cd in zip(data["cls"], data["cds"])]
        idx = np.argmax(l_over_d)
        return {
            "alpha_opt": math.radians(data["alphas"][idx]),
            "cl_opt": data["cls"][idx]
        }

    def get_cl_cd(self, alpha_rad):
        """Evaluates the interpolators using radians."""
        alpha_deg = math.degrees(alpha_rad)
        interps = self.polar_interpolators
        return float(interps["cl"](alpha_deg)), float(interps["cd"](alpha_deg))


# --- 2. SECTIONAL PHYSICS ---
class BladeSection(Base):
    """The core BEMT solver for a single radial station."""
    radius = Input()
    dr = Input()
    chord = Input()
    pitch = Input()
    total_radius = Input()
    n_blades = Input()
    rpm = Input()
    air_density = Input(1.225)
    airfoil_obj = Input()
    target_thrust = Input()

    @Attribute
    def induced_velocity_ideal(self):
        """Ideal uniform induced velocity from 1D momentum theory[cite: 28]."""
        disk_area = math.pi * self.total_radius ** 2
        return math.sqrt(self.target_thrust / (2.0 * self.air_density * disk_area))

    @Attribute
    def optimum_geometry(self):
        """Generates geometry to hit ideal flow state [cite: 74-78, 114]."""
        omega = self.rpm * 2.0 * math.pi / 60.0
        vi = self.induced_velocity_ideal
        v_ax = vi
        v_rot = omega * self.radius
        v_eff = math.sqrt(v_ax ** 2 + v_rot ** 2)
        phi = math.atan2(v_ax, v_rot)

        # Optimum Pitch Rule [cite: 75]
        alpha_opt = self.airfoil_obj.optimal_performance["alpha_opt"]
        pitch = phi + alpha_opt

        # Prandtl Tip-Loss Factor [cite: 56-57]
        f_tip = (self.n_blades / 2.0) * (self.total_radius - self.radius) / max(1e-6, self.radius * math.sin(phi))
        F = (2.0 / math.pi) * math.acos(max(0.0, min(1.0, math.exp(-f_tip))))

        # Optimum Chord Rule [cite: 78]
        cl_opt = self.airfoil_obj.optimal_performance["cl_opt"]
        chord = (8.0 * math.pi * self.radius * vi ** 2 * F) / (self.n_blades * v_eff ** 2 * cl_opt * math.cos(phi))

        return {"chord": max(0.005, chord), "pitch": pitch, "phi": phi, "F": F}

    @Attribute
    def aerodynamics(self):
        """Calculates actual sectional forces based on generated geometry [cite: 45-48]."""
        geom = self.optimum_geometry
        omega = self.rpm * 2.0 * math.pi / 60.0
        v_ax = self.induced_velocity_ideal
        v_rot = omega * self.radius
        v_eff = math.sqrt(v_ax ** 2 + v_rot ** 2)

        cl, cd = self.airfoil_obj.get_cl_cd(self.airfoil_obj.optimal_performance["alpha_opt"])

        l_prime = 0.5 * self.air_density * v_eff ** 2 * geom["chord"] * cl
        d_prime = 0.5 * self.air_density * v_eff ** 2 * geom["chord"] * cd

        dT = (l_prime * math.cos(geom["phi"]) - d_prime * math.sin(geom["phi"])) * self.n_blades * self.dr
        dQ = (l_prime * math.sin(geom["phi"]) + d_prime * math.cos(geom["phi"])) * self.radius * self.n_blades * self.dr

        return {"dT": dT, "dQ": dQ}

# --- 3. PROPELLER ASSEMBLY ---
class HoverPropeller(Base):
    """Aggregates sections into a complete propeller design."""
    diameter = Input(0.4)
    hub_diameter = Input(0.04)
    n_blades = Input(2)
    rpm = Input(5000)
    n_segments = Input(15)
    chord = Input(0.03)
    pitch = Input(math.radians(15))
    target_thrust = Input()

    @Part
    def airfoil(self):
        return Airfoil(naca_code="4412")

    @Attribute
    def dr(self):
        return (self.diameter / 2 - self.hub_diameter / 2) / self.n_segments

    @Part
    def sections(self):
        return Sequence(
            type=BladeSection,
            quantify=self.n_segments,
            radius=lambda item: (self.hub_diameter / 2) + (item.index + 0.5) * self.dr,
            dr=self.dr,
            chord=self.chord,
            pitch=self.pitch,
            total_radius=self.diameter / 2,
            n_blades=self.n_blades,
            rpm=self.rpm,
            airfoil_obj=self.airfoil
        )

    @Attribute
    def total_thrust(self):
        return sum(s.aerodynamics["dT"] for s in self.sections)

    @Attribute
    def total_torque(self):
        return sum(s.aerodynamics["dQ"] for s in self.sections)


# --- 4. POWERTRAIN MATCHING ---
class ElectricMotor(Base):
    """Checks if a motor can handle the propeller's power requirements."""
    kv = Input(900)
    max_power = Input(500)
    max_current = Input(30)
    torque_req = Input()
    rpm_req = Input()

    @Attribute
    def is_feasible(self):
        # Kt (Torque Constant) is roughly 1/Kv in SI units
        kt = 1 / (self.kv * 2 * math.pi / 60)
        current_draw = self.torque_req / kt
        power_draw = current_draw * (self.rpm_req / self.kv)

        return current_draw <= self.max_current and power_draw <= self.max_power


# --- 5. TOP LEVEL SYSTEM ---
class PropulsionSystem(Base):
    """The master class for the entire vehicle powertrain."""
    mtow = Input(2.0)
    n_rotors = Input(4)
    safety_margin = Input(1.5)

    @Attribute
    def thrust_required(self):
        return (self.mtow * 9.81 / self.n_rotors) * self.safety_margin

    @Part
    def propeller(self):
        return HoverPropeller(rpm=5000, target_thrust = self.thrust_required)

    @Part
    def motor(self):
        return ElectricMotor(
            torque_req=self.propeller.total_torque,
            rpm_req=self.propeller.rpm
        )

if __name__ == '__main__':
    from parapy.gui import display
    airfoil = Airfoil()
    polars=airfoil.polar_interpolators
    print(polars)
    display(PropulsionSystem())