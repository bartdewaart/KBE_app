import math

import numpy as np
from parapy.core import Base, Input, Attribute, Part, Sequence
from parapy.geom import Cylinder, Vector
from scipy.interpolate import CubicSpline, PchipInterpolator

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
    hub_radius = Input(0.02)  # increased default for realistic hub

    #: optional input slot - hub height [m]
    hub_height = Input(0.04)

    #: optional input slot - geometry limits
    min_chord = Input(0.004)
    max_chord_fraction = Input(0.30)
    min_pitch_deg = Input(2.0)
    max_pitch_deg = Input(45.0)

    #: optional input slot - inner section cutoff (fraction of span)
    root_cutoff_ratio = Input(0.10)


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
        return ((self.base_thrust
             + self.estimated_mass_design * 9.81)
            * self.safety_margin)

    @Attribute
    def splines(self):
        """
        Generative Rule: computes optimal chord and pitch distributions
        at 5 control points using Betz momentum theory, then fits
        CubicSpline objects for smooth interpolation across all sections.
        """
        r_hub  = self.hub_radius
        r_tip  = self.diameter / 2
        r_ctrl = np.linspace(r_hub, r_tip, 11)

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
        min_pitch = math.radians(self.min_pitch_deg)
        max_pitch = math.radians(self.max_pitch_deg)

        max_chord = max(self.min_chord, self.max_chord_fraction * r_tip)

        def _opt_chord_pitch(r):
            phi = math.atan2(vi, omega * r)
            v_eff = math.sqrt(vi ** 2 + (omega * r) ** 2)

            f_tip = ((self.n_blades / 2)
                     * (r_tip - r) / max(1e-6, r * math.sin(phi)))
            F = ((2 / math.pi)
                  * math.acos(max(0.0, min(1.0, math.exp(-f_tip)))))
            F = max(0.05, F)

            chord = ((8 * math.pi * r * vi ** 2 * F)
                     / (self.n_blades
                        * v_eff ** 2
                        * self.airfoil.polar_data["cl_opt"]
                        * math.cos(phi)))
            if not math.isfinite(chord) or chord <= 0:
                chord = self.min_chord

            pitch = phi + self.airfoil.polar_data["alpha_opt_rad"]
            pitch = max(min_pitch, min(max_pitch, pitch))

            chord = max(self.min_chord, min(max_chord, chord))
            return chord, pitch

        root_cutoff = max(0.0, min(0.6, self.root_cutoff_ratio))
        r_root = r_hub + root_cutoff * (r_tip - r_hub)
        root_chord, root_pitch = _opt_chord_pitch(r_root)
        chord_hub = min(max_chord, max(self.min_chord, 2.0 * self.hub_radius))
        pitch_hub = max(root_pitch, min_pitch)
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
            F = max(0.05, F)

            # Optimum Chord/Pitch Generation
            chord, pitch = _opt_chord_pitch(r)
            if chord <= self.min_chord:
                print(f"WARNING: Chord clamped to minimum at r={r:.3f}m "
                      f"(v_eff={v_eff:.3f}, phi={math.degrees(phi):.1f}deg)")

            # Inner section transition to structural hub profile
            if r < r_root:
                if r_root > r_hub:
                    t = (r - r_hub) / (r_root - r_hub)
                else:
                    t = 0.0
                chord = chord_hub + t * (root_chord - chord_hub)
                pitch = pitch_hub + t * (root_pitch - pitch_hub)

            c_ctrl.append(max(self.min_chord, chord))
            p_ctrl.append(pitch)

        if getattr(self, "debug_splines", True):
            print("DEBUG chord control points (r, chord):")
            for r, c in zip(r_ctrl, c_ctrl):
                print(f"  r={r:.3f} m, c={c:.4f} m")
            r_samples = np.linspace(r_hub, r_tip, 9)
            c_spline = CubicSpline(r_ctrl, c_ctrl)
            print("DEBUG chord samples:")
            for r in r_samples:
                print(f"  r={r:.3f} m, c={float(c_spline(r)):.4f} m")

        return CubicSpline(r_ctrl, c_ctrl), PchipInterpolator(r_ctrl, p_ctrl)

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
        # Unit-chord airfoil area (x,y) from the generated coordinates.
        pts = [(p[0], p[1]) for p in self.airfoil.points]
        area_unit = 0.0
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            area_unit += x0 * y1 - x1 * y0
        area_unit = abs(area_unit) * 0.5

        blade_volume = sum(
            area_unit * (s.chord ** 2) * s.dr
            for blade in self.blades
            for s in blade.sections
        )
        return blade_volume * density_material

    @Part
    def hub(self):
        """
        Geometry Rule: cylindrical hub placeholder at rotor centre.
        """
        return Cylinder(
            radius=self.hub_radius,
            height=self.hub_height,
            centered=True,
            color="DarkSlateGray"
        )