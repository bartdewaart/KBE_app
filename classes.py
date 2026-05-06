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
        polar_file = f"polar_{self.naca_code}_{int(self.reynolds)}.txt"

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

        # Parsing logic to build Scipy interpolators
        alphas, cls, cds = [], [], []
        if os.path.exists(polar_file):
            with open(polar_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3 and parts[0][0].isdigit() or parts[0][0] == '-':
                        alphas.append(float(parts[0]))
                        cls.append(float(parts[1]))
                        cds.append(float(parts[2]))

        cl_interp = interp1d(alphas, cls, kind='linear', fill_value="extrapolate")
        cd_interp = interp1d(alphas, cds, kind='linear', fill_value="extrapolate")
        return {"cl": cl_interp, "cd": cd_interp}

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

    @Attribute
    def aerodynamics(self):
        """Iteratively solves for local thrust (dT) and torque (dQ)."""
        omega = self.rpm * 2.0 * math.pi / 60.0
        v_ax = 0.0  # Initial axial velocity guess
        v_rot = omega * self.radius

        # Simplified BEMT loop logic
        v_eff = math.sqrt(v_ax ** 2 + v_rot ** 2)
        phi = math.atan2(v_ax, v_rot)
        alpha = self.pitch - phi

        cl, cd = self.airfoil_obj.get_cl_cd(alpha)

        # Force calculations
        l_prime = 0.5 * self.air_density * v_eff ** 2 * self.chord * cl
        d_prime = 0.5 * self.air_density * v_eff ** 2 * self.chord * cd

        dT = (l_prime * math.cos(phi) - d_prime * math.sin(phi)) * self.n_blades * self.dr
        dQ = (l_prime * math.sin(phi) + d_prime * math.cos(phi)) * self.radius * self.n_blades * self.dr

        return {"dT": dT, "dQ": dQ}


# --- 3. PROPELLER ASSEMBLY ---
class HoverPropeller(Base):
    """Aggregates sections into a complete propeller design."""
    diameter = Input(0.4)
    hub_diameter = Input(0.04)
    n_blades = Input(2)
    rpm = Input(5000)
    n_segments = Input(15)

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
            quantifier=self.n_segments,
            radius=lambda item: (self.hub_diameter / 2) + (item.index + 0.5) * self.dr,
            dr=self.dr,
            chord=0.03,
            pitch=math.radians(15),
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
        return HoverPropeller(rpm=5000)

    @Part
    def motor(self):
        return ElectricMotor(
            torque_req=self.propeller.total_torque,
            rpm_req=self.propeller.rpm
        )