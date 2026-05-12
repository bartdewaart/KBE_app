import math

import numpy as np
from parapy.core import Base, Input, Attribute, Part, Sequence
from parapy.geom import Cylinder, Vector
from scipy.interpolate import CubicSpline

from .airfoil import Airfoil
from .blade import Blade


class Propeller(Base):
    """
    Main propeller assembly class.
    Generates the optimal blade geometry using Betz momentum theory,
    assembles the full rotor from n_blades Blade objects, and
    aggregates total aerodynamic performance.
    """

    #: required input slot — propeller diameter [m]
    diameter = Input()

    #: required input slot — rotational speed [RPM]
    rpm = Input()

    #: optional input slot - number of blades
    n_blades = Input(2)

    #: optional input slot - NACA airfoil code e.g. '4412'
    airfoil_type = Input("4412")

    #: required input slot - base thrust requirement per rotor [N]
    #: excludes rotor self-weight (added internally via design_thrust)
    base_thrust = Input()

    #: required input slot - safety margin on thrust
    #: e.g. 1.5 means rotor must produce 150% of base thrust
    safety_margin = Input()

    #: optional input slot - number of spanwise analysis sections
    n_segments = Input(30)

    #: optional input slot - air density [kg/m³]
    air_density = Input(1.225)

    #: optional input slot - hub radius [m], default value of 8mm
    hub_radius = Input(0.008)  # add this

    @Part(parse=False)
    def airfoil(self):
        """
        Integration Rule: instantiates the shared Airfoil object.
        Triggers XFOIL polar generation for this airfoil type.
        All blades and sections share this single Airfoil instance.
        """
        return Airfoil(
            naca_code=self.airfoil_type,
            reynolds=300000
        )

    @Attribute
    def estimated_mass_design(self):
        """
        Mathematical Rule: simplified blade mass estimate used to
        break the circular dependency between mass and target_thrust.
        Uses a heuristic volume approximation with carbon fibre density.
        """
        r_tip      = self.diameter / 2
        # Heuristic: n_blades * (span * avg_chord * avg_thickness)
        vol_approx = self.n_blades * (r_tip * 0.05 * 0.005)
        return vol_approx * 1600  # Carbon fibre density [kg/m³]

    @Attribute
    def design_thrust(self):
        """
        Mathematical Rule: thrust target used for spline generation.
        Uses estimated mass to avoid circular dependency with actual mass.
        """
        return (self.base_thrust
                + self.estimated_mass_design * 9.81 * self.safety_margin)

    @Attribute
    def splines(self):
        """
        Generative Rule: computes optimal chord and pitch distributions
        at 5 control points using Betz momentum theory, then fits
        CubicSpline objects for smooth interpolation across all sections.
        """
        r_hub  = self.hub_radius
        r_tip  = self.diameter / 2
        r_ctrl = np.linspace(r_hub, r_tip, 5)

        # Mathematical Rule: uniform induced velocity from 1D momentum theory
        vi    = math.sqrt(
            self.design_thrust / (2.0 * self.air_density * math.pi * r_tip ** 2)
        )
        omega = self.rpm * 2 * math.pi / 60

        if vi <= 0 or not math.isfinite(vi):
            print(
                f"WARNING: Induced velocity vi={vi:.4f} m/s is invalid. "
                f"Clamping to minimum. Check design_thrust and diameter."
            )
            vi = 0.1  # minimum fallback

        c_ctrl, p_ctrl = [], []
        for r in r_ctrl:
            phi   = math.atan2(vi, omega * r)
            v_eff = math.sqrt(vi ** 2 + (omega * r) ** 2)

            # Prandtl Tip-Loss Factor at control point
            f_tip = ((self.n_blades / 2)
                     * (r_tip - r) / max(1e-6, r * math.sin(phi)))
            F     = ((2 / math.pi)
                     * math.acos(max(0.0, min(1.0, math.exp(-f_tip)))))
            # Enforce minimum tip-loss correction to avoid chord collapse
            # When f_tip is very small, F approaches 0 which zeros out chord
            F = max(0.4, F)

            # Optimum Chord Generation
            chord = ((8 * math.pi * r * vi ** 2 * F)
                     / (self.n_blades
                        * v_eff ** 2
                        * self.airfoil.polar_data["cl_opt"]
                        * math.cos(phi)))

            if not math.isfinite(chord) or chord <= 0:
                chord = 0.02  # minimum chord — clamp instead of crash
                print(f"WARNING: Chord clamped to minimum at r={r:.3f}m "
                      f"(v_eff={v_eff:.3f}, phi={math.degrees(phi):.1f}deg)")
            else:
                chord = max(0.02, chord)

            # Optimum Pitch Generation
            c_ctrl.append(max(0.02, chord))
            p_ctrl.append(phi + self.airfoil.polar_data["alpha_opt_rad"])

        return CubicSpline(r_ctrl, c_ctrl), CubicSpline(r_ctrl, p_ctrl)

    @Part
    def blades(self):
        """
        Configuration Rule: instantiates n_blades Blade objects,
        each rotated evenly around the hub Z-axis.
        Rotation angle = index * (2π / n_blades).
        """
        return Sequence(
        type=Blade,
        quantify=self.n_blades,
        n_segments=self.n_segments,
        rotation_angle=lambda child: child.index * (2 * math.pi / self.n_blades)
    )

    @Attribute
    def total_thrust(self):
        """
        Mathematical Rule: total rotor thrust [N].
        Aggregates blade_thrust from all Blade objects.
        """
        thrust = sum(blade.blade_thrust for blade in self.blades)
        if thrust < 0:
            print(
                f"WARNING: Total thrust is negative ({thrust:.2f} N). "
                f"This usually means pitch angles are inverted or RPM "
                f"is too low. Check spline pitch distribution."
            )
        return thrust

    @Attribute
    def total_torque(self):
        """
        Mathematical Rule: total rotor torque [Nm].
        Aggregates blade_torque from all Blade objects.
        """
        torque = sum(blade.blade_torque for blade in self.blades)
        if torque < 0:
            print(
                f"WARNING: Total torque is negative ({torque:.2f} Nm). "
                f"Check spline pitch distribution."
            )
        return torque

    @Attribute
    def performance(self):
        """
        Mathematical Rule: combined rotor performance summary.
        Returns thrust [N], torque [Nm] and shaft power [W].
        """
        omega       = self.rpm * 2 * math.pi / 60
        shaft_power = self.total_torque * omega
        return {
            "thrust"      : self.total_thrust,
            "torque"      : self.total_torque,
            "shaft_power" : shaft_power
        }

    @Attribute
    def mass(self):
        """
        Mathematical Rule: total rotor mass based on actual blade
        geometry. Sums cross-sectional volume of each section and
        applies carbon fibre density.
        """
        density_material = 1600  # Carbon fibre [kg/m³]
        blade_volume = sum(
            s.chord * (s.chord * 0.12) * s.dr
            for blade in self.blades
            for s in blade.sections
        )
        return blade_volume * density_material # removed n_blades multiplier (duplicity)

    @Part
    def hub(self):
        """
        Geometry Rule: cylindrical hub placeholder at rotor centre.
        """
        return Cylinder(
            radius=0.02,
            height=0.04,
            centered=True,
            color="DarkSlateGray"
        )