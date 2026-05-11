import math
import numpy as np
from parapy.core import Base, Input, Attribute, Part, Sequence
from parapy.geom import Cylinder, Point, RotatedShape, Vector
from scipy.interpolate import CubicSpline
from .blade import Blade
from .config import get_value

class Propeller(Base):
    diameter = Input(get_value("propeller", "diameter", default=0.3))
    rpm = Input(get_value("propeller", "rpm", default=5000))
    n_blades = Input(get_value("propeller", "n_blades", default=2))
    airfoil_type = Input(get_value("propeller", "airfoil_type", default="4412"))
    base_thrust = Input()
    n_segments = Input(get_value("propeller", "n_segments", default=20))
    thrust_to_weight = Input(get_value("propeller", "thrust_to_weight", default=1.5))
    min_chord = Input(get_value("propeller", "min_chord", default=0.005))
    hub_radius = Input(get_value("propeller", "hub_radius", default=0.02))
    hub_height = Input(get_value("propeller", "hub_height", default=0.04))
    pitch_offset = Input(get_value("propeller", "pitch_offset", default=0.05))
    pitch_sign = Input(get_value("propeller", "pitch_sign", default=1.0))
    cl_opt = Input(get_value("propeller", "cl_opt", default=1.0))

    @Attribute
    def splines(self):
        r_hub, r_tip = self.hub_radius, self.diameter / 2
        r_ctrl = np.linspace(r_hub, r_tip, 5)
        # BEMT Idealization logic
        vi = math.sqrt(self.design_thrust / (2.0 * 1.225 * math.pi * r_tip ** 2))
        omega = self.rpm * math.pi / 30
        c_ctrl = []
        p_ctrl = []
        for r in r_ctrl:
            phi = math.atan2(vi, omega * r)
            sin_phi = max(1e-6, math.sin(phi))
            f_tip = (self.n_blades / 2.0) * (r_tip - r) / (r * sin_phi)
            F = (2.0 / math.pi) * math.acos(math.exp(-f_tip))
            F = min(1.0, max(1e-3, F))

            v_eff_sq = vi ** 2 + (omega * r) ** 2
            cos_phi = max(1e-3, math.cos(phi))
            chord = (8.0 * math.pi * r * vi ** 2 * F) / (self.n_blades * v_eff_sq * self.cl_opt * cos_phi)
            c_ctrl.append(max(self.min_chord, chord))
            pitch_rad = self.pitch_sign * (phi + self.pitch_offset)
            p_ctrl.append(pitch_rad)
        return CubicSpline(r_ctrl, c_ctrl), CubicSpline(r_ctrl, p_ctrl)

    @Part
    def hub(self):
        return Cylinder(radius=self.hub_radius, height=self.hub_height, centered=True, color="Gray")

    @Part
    def blade_geometry(self):
        """Single 3D blade."""
        return Blade(n_segments=self.n_segments)

    @Part(parse=False)
    def all_blades(self):
        """Circular pattern."""
        return [
            RotatedShape(
                shape_in=self.blade_geometry,
                angle=i * (2 * math.pi / self.n_blades),
                vector=Vector(0, 0, 1),
                rotation_point=Point(0, 0, 0)
            )
            for i in range(self.n_blades)
        ]

    @Attribute
    def performance(self):
        sects = self.blade_geometry.sections
        return {"thrust": sum(s.aerodynamics["dT"] for s in sects),
                "torque": sum(s.aerodynamics["dQ"] for s in sects)}

    @Attribute
    def mass(self):
        sects = self.blade_geometry.sections
        vol = sum(s.chord * (s.chord * 0.12) * s.dr for s in sects)
        return self.n_blades * vol * 1600

    @Attribute
    def design_thrust(self):
        """Estimate to break circular dependency."""
        m_est = self.n_blades * (self.diameter/2 * 0.05 * 0.005) * 1600
        return self.base_thrust + (m_est * 9.81 * self.thrust_to_weight)

    @Attribute
    def target_thrust(self):
        """
        The actual thrust requirement based on the final calculated mass.
        Used by the optimizer's constraint function.
        """
        return self.base_thrust + (self.mass * 9.81 * self.thrust_to_weight)