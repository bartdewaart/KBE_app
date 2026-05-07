import math
from parapy.core import Base, Input, Attribute, Part, Sequence
from .airfoil import Airfoil
from .blade_section import BladeSection
from scipy.interpolate import CubicSpline
import numpy as np
from parapy.geom import Cylinder, RotatedShape, Vector
from .blade import Blade


class Propeller(Base):
    diameter = Input()
    rpm = Input()
    n_blades = Input()
    airfoil_type = Input()
    base_thrust = Input()
    n_segments = Input(30)
    thrust_to_weight=Input()

    @Attribute
    def airfoil(self): return Airfoil(naca_code=self.airfoil_type, reynolds=30000)

    @Attribute
    def estimated_mass_design(self):
        """
        Simplified mass for breaking circular dependency.
        Estimates volume as a fraction of the disk area.
        """
        # Heuristic: ~5% of a thin disk volume with material density
        r_tip = self.diameter / 2
        vol_approx = self.n_blades * (r_tip * 0.05 * 0.005)  # length * avg_chord * avg_thick
        return vol_approx * 1600  # Carbon Fiber density

    @Attribute
    def design_thrust(self):
        """Thrust used for spline generation (uses Estimate)."""
        return self.base_thrust + (self.estimated_mass_design * 9.81 * self.thrust_to_weight)

    @Attribute
    def target_thrust(self):
        """Strict requirement for the optimizer (uses Actual Mass)."""
        return self.base_thrust + (self.mass * 9.81 * self.thrust_to_weight)

    @Attribute
    def splines(self):
        """Generative Rule: Creates smooth chord and pitch distributions."""
        r_hub, r_tip = 0.02, self.diameter / 2
        r_ctrl = np.linspace(r_hub, r_tip, 5)

        # Calculate ideal values at control points using Betz
        vi = math.sqrt(self.design_thrust / (2.0 * 1.225 * math.pi * r_tip ** 2))
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

    @Part
    def sections(self):
        return Sequence(
            type=BladeSection,
            quantify=self.n_segments
        )

    @Attribute
    def performance(self):
        sects = self.sections
        return {"thrust": sum(s.aerodynamics["dT"] for s in sects),
                "torque": sum(s.aerodynamics["dQ"] for s in sects)}

    @Attribute
    def mass(self):
        """Final accurate mass based on actual section geometry."""
        density_material = 1600
        # This sums the volume of the individual sections
        blade_volume = sum(s.chord * (s.chord * 0.12) * s.dr for s in self.sections)
        return self.n_blades * blade_volume * density_material

    @Part
    def hub(self):
        return Cylinder(radius=0.02, height=0.04, centered=True, color="DarkSlateGray")

    # @Part
    # def blade_geom(self):
    #     """The single blade instance."""
    #     return Blade(n_segments=self.n_segments)

    # @Part
    # def all_blades(self):
    #     """Circular pattern of the lofted blade."""
    #     return Sequence(type=RotatedShape,
    #                     quantify=self.n_blades,
    #                     shape_in=self.blade_geom.surface,
    #                     angle=lambda item: item.index * (2 * math.pi / self.n_blades),
    #                     vector=Vector(0, 0, 1))