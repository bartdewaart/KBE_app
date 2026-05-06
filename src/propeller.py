import math
from parapy.core import Base, Input, Attribute, Part, Sequence
from .airfoil import Airfoil
from .blade_section import BladeSection
from scipy.interpolate import CubicSpline
import numpy as np


class Propeller(Base):
    diameter = Input()
    rpm = Input()
    n_blades = Input()
    airfoil_type = Input()
    target_thrust = Input()
    n_segments = Input(30)

    @Part
    def airfoil(self): return Airfoil(naca_code=self.airfoil_type)

    @Attribute
    def splines(self):
        """Generative Rule: Creates smooth chord and pitch distributions."""
        r_hub, r_tip = 0.02, self.diameter / 2
        r_ctrl = np.linspace(r_hub, r_tip, 5)

        # Calculate ideal values at control points using Betz
        vi = math.sqrt(self.target_thrust / (2.0 * 1.225 * math.pi * r_tip ** 2))
        omega = self.rpm * 2 * math.pi / 60

        c_ctrl, p_ctrl = [], []
        for r in r_ctrl:
            phi = math.atan2(vi, omega * r)
            v_eff = math.sqrt(vi ** 2 + (omega * r) ** 2)
            f_tip = (self.n_blades / 2) * (r_tip - r) / max(1e-6, r * math.sin(phi))
            F = (2 / math.pi) * math.acos(max(0.0, min(1.0, math.exp(-f_tip))))

            chord = (8 * math.pi * r * vi ** 2 * F) / (
                        self.n_blades * v_eff ** 2 * self.airfoil.polar_data["cl_opt"] * math.cos(phi))
            c_ctrl.append(max(0.02, chord))
            p_ctrl.append(phi + self.airfoil.polar_data["alpha_opt_rad"])

        return CubicSpline(r_ctrl, c_ctrl), CubicSpline(r_ctrl, p_ctrl)

    @Part(parse=False)
    def sections(self):
        return Sequence(
            type=BladeSection,
            quantify=self.n_segments
        )

    @Attribute
    def performance(self):
        return {"thrust": sum(s.aerodynamics["dT"] for s in self.sections),
                "torque": sum(s.aerodynamics["dQ"] for s in self.sections)}