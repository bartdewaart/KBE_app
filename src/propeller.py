import math

import numpy as np
from parapy.core import Base, Input, Attribute, Part, Sequence
from parapy.geom import Cylinder, Point, Vector, translate
from scipy.interpolate import CubicSpline, PchipInterpolator

from .airfoil import Airfoil
from .blade import Blade


class Propeller(Base):
    """
    Main propeller assembly class.

    Generates the optimal blade geometry using Betz momentum theory,
    assembles the full rotor from n_blades Blade objects, and
    aggregates total aerodynamic performance.

    Design flow:
        1. splines  — computes Betz-optimal chord/pitch at 11 control
                      points and fits CubicSpline / PchipInterpolator.
        2. blades   — instantiates n_blades Blade objects; each owns a
                      spanwise sequence of BladeSection objects that run
                      BEMT and build 3D geometry.
        3. Performance attributes aggregate per-blade BEMT results.
    """

    # ─── Inputs ──────────────────────────────────────────────────────────────

    #: required input slot — propeller diameter [m]
    diameter = Input()

    #: required input slot — rotational speed [RPM]
    rpm = Input()

    #: optional input slot — number of blades
    n_blades = Input(2)

    #: optional input slot — NACA airfoil code, e.g. '4412'
    airfoil_type = Input("4412")

    #: required input slot — base thrust requirement per rotor [N]
    #: excludes rotor self-weight (added internally via design_thrust)
    base_thrust = Input()

    #: required input slot — safety margin on thrust
    #: e.g. 1.5 means rotor must produce 150% of base thrust
    safety_margin = Input()

    #: optional input slot — number of spanwise analysis sections
    n_segments = Input(30)

    #: optional input slot — air density [kg/m³]
    air_density = Input(1.225)

    #: optional input slot — hub radius [m]
    hub_radius = Input(0.02)

    #: optional input slot — hub height [m]
    hub_height = Input(0.04)

    #: optional input slot — geometry limits
    min_chord = Input(0.004)
    #: optional input slot — minimum chord as a fraction of blade span [r_tip − r_hub].
    #: Prevents the Betz formula from collapsing chord to the absolute min_chord
    #: at high RPM.  Effective minimum = max(min_chord, min_chord_fraction × span).
    #: 0.10 gives ~18 mm for a 0.4 m prop — close to real DJI-class chord widths.
    min_chord_fraction = Input(0.10)
    max_chord_fraction = Input(0.30)
    min_pitch_deg = Input(2.0)
    max_pitch_deg = Input(60.0)

    #: optional input slot — inner section cutoff (fraction of span)
    #: sections inboard of this are treated as structural, not aero
    root_cutoff_ratio = Input(0.10)

    #: optional input slot — outer section cutoff (fraction of span)
    #: sections outboard of this are skipped by BEMT to avoid the
    #: Prandtl tip-loss singularity (F → 0)
    tip_cutoff_ratio = Input(0.02)

    #: optional input slot — fraction of span over which to apply a
    #: cosine tip-relief that rounds the planform near the tip.
    #: 0.0 disables (hard-clamped tip); 0.15 = moderate; 0.30 = pronounced.
    tip_relief_ratio = Input(0.15)

    #: optional input slot — print per-section BEMT diagnostic warnings
    #: keep False during optimisation to avoid console flood
    verbose = Input(False)

    #: optional input slot — print spline chord-table debug output
    debug_splines = Input(False)

    #: optional input slot — blade material density [kg/m³]
    #: propagated from PropulsionSystem.material_density so the GUI
    #: material dropdown flows through to both mass attributes here.
    material_density = Input(1600)

    #: optional input slot — BEMT under-relaxation factor [-]
    bemt_relaxation = Input(0.3)

    #: optional input slot — BEMT convergence tolerance [m/s]
    bemt_tolerance = Input(1e-4)

    #: optional input slot — BEMT maximum iterations [-]
    bemt_max_iter = Input(100)

    #: optional input slot — kinematic viscosity of air [m²/s]
    #: used for representative Reynolds number calculation; 1.5e-5 is standard sea level ~20 °C
    air_kinematic_viscosity = Input(1.5e-5)

    # ─── Airfoil ─────────────────────────────────────────────────────────────

    @Attribute
    def representative_reynolds(self):
        """
        Estimated Re at 75% span for XFOIL polar generation.

        Uses V_eff(75% span) and the min-chord floor as a conservative chord
        estimate — the floor avoids circularity (polar not yet known).
        Rounded to the nearest 50 000 so small optimizer steps in diameter or
        RPM don't invalidate the cached polar and re-trigger XFOIL.
        """
        r_tip  = self.diameter / 2
        r_75   = self.hub_radius + 0.75 * (r_tip - self.hub_radius)
        omega  = self.rpm * 2 * math.pi / 60
        vi     = math.sqrt(
            max(self.design_thrust, 1e-6)
            / (2 * self.air_density * math.pi * max(r_tip ** 2, 1e-9))
        )
        v_eff  = math.sqrt(vi ** 2 + (omega * r_75) ** 2)
        chord  = max(self.min_chord, self.min_chord_fraction * (r_tip - self.hub_radius))
        re     = v_eff * chord / self.air_kinematic_viscosity
        return max(50000, round(re / 50000) * 50000)

    @Part(parse=False)
    def airfoil(self):
        """
        Integration Rule: shared Airfoil object for this propeller.
        Triggers XFOIL polar generation once; all blades and sections
        reference this single instance.
        """
        return Airfoil(
            naca_code=self.airfoil_type,
            reynolds=self.representative_reynolds,
        )

    # ─── Design mass / thrust targets ────────────────────────────────────────

    @Attribute
    def estimated_mass_design(self):
        """
        Mathematical Rule: simplified blade mass estimate used to break
        the circular dependency between rotor mass and target thrust.
        Uses a heuristic volume approximation with carbon-fibre density.
        """
        r_tip      = self.diameter / 2
        vol_approx = self.n_blades * (r_tip * 0.05 * 0.005)
        return vol_approx * self.material_density

    @Attribute
    def design_thrust(self):
        """
        Mathematical Rule: thrust target used for spline generation.
        Uses the estimated (heuristic) mass to break the circularity
        between actual mass and required thrust.
        """
        return (
            (self.base_thrust + self.estimated_mass_design * 9.81)
            * self.safety_margin
        )

    # ─── Betz-optimal chord/pitch splines ────────────────────────────────────

    @Attribute
    def splines(self):
        """
        Generative Rule: computes optimal chord and pitch at 11 control
        points using Betz momentum theory, then fits CubicSpline (chord)
        and PchipInterpolator (pitch) for smooth spanwise interpolation.

        Control-point recipe:
        1. Uniform induced velocity from 1-D momentum theory.
        2. For each r: Prandtl tip-loss factor F; Betz-optimal chord and
           pitch from inflow angle φ = atan2(v_i, ω r).
        3. Inboard of r_root: linear blend from a full-chord hub profile
           to the first aerodynamic section (structural transition).
        4. Cosine tip-relief post-process: narrow the outermost control
           points to follow a quarter-cosine from c_ref down to min_chord
           at the tip.  The post-process only *narrows* (min operator) so
           it never widens the planform, keeping the design conservative.
        """
        r_hub  = self.hub_radius
        r_tip  = self.diameter / 2
        r_ctrl = np.linspace(r_hub, r_tip, 11)

        # 1-D momentum theory: uniform hover induced velocity
        vi = math.sqrt(
            self.design_thrust / (2.0 * self.air_density * math.pi * r_tip ** 2)
        )
        omega = self.rpm * 2 * math.pi / 60

        if vi <= 0 or not math.isfinite(vi):
            print(
                f"WARNING: Induced velocity vi={vi:.4f} m/s is invalid. "
                f"Clamping to minimum. Check design_thrust and diameter."
            )
            vi = 0.1

        min_pitch = math.radians(self.min_pitch_deg)
        max_pitch = math.radians(self.max_pitch_deg)
        max_chord = max(self.min_chord, self.max_chord_fraction * r_tip)
        # Span-relative minimum chord prevents Betz formula from collapsing at high RPM.
        eff_min_chord = max(self.min_chord, self.min_chord_fraction * (r_tip - r_hub))

        def _opt_chord_pitch(r):
            """Return Betz-optimal (chord, pitch) at radial station r."""
            phi   = math.atan2(vi, omega * r)
            v_eff = math.sqrt(vi ** 2 + (omega * r) ** 2)

            f_tip = (self.n_blades / 2) * (r_tip - r) / max(1e-6, r * math.sin(phi))
            F     = (2 / math.pi) * math.acos(max(0.0, min(1.0, math.exp(-f_tip))))
            F     = max(0.05, F)  # floor avoids chord collapsing to zero at tip

            chord = (
                (8 * math.pi * r * vi ** 2 * F)
                / (self.n_blades
                   * v_eff ** 2
                   * self.airfoil.polar_data["cl_opt"]
                   * math.cos(phi))
            )
            if not math.isfinite(chord) or chord <= 0:
                chord = eff_min_chord

            pitch = phi + self.airfoil.polar_data["alpha_opt_rad"]
            pitch = max(min_pitch, min(max_pitch, pitch))
            chord = max(eff_min_chord, min(max_chord, chord))
            return chord, pitch

        # Hub-to-root structural blend: fix chord/pitch at hub face and
        # interpolate linearly to the first aerodynamic section at r_root.
        root_cutoff = max(0.0, min(0.6, self.root_cutoff_ratio))
        r_root      = r_hub + root_cutoff * (r_tip - r_hub)
        root_chord, root_pitch = _opt_chord_pitch(r_root)
        chord_hub   = min(max_chord, max(eff_min_chord, 2.0 * self.hub_radius))
        pitch_hub   = max(root_pitch, min_pitch)

        c_ctrl, p_ctrl = [], []
        for r in r_ctrl:
            chord, pitch = _opt_chord_pitch(r)

            if self.debug_splines and chord <= eff_min_chord:
                # Recompute phi/v_eff locally — only needed for the message.
                phi   = math.atan2(vi, omega * r)
                v_eff = math.sqrt(vi ** 2 + (omega * r) ** 2)
                print(
                    f"DEBUG: chord clamped to min at r={r:.3f} m "
                    f"(v_eff={v_eff:.3f}, phi={math.degrees(phi):.1f} deg)"
                )

            if r < r_root:
                t = (r - r_hub) / (r_root - r_hub) if r_root > r_hub else 0.0
                chord = chord_hub + t * (root_chord - chord_hub)
                pitch = pitch_hub + t * (root_pitch - pitch_hub)

            c_ctrl.append(max(eff_min_chord, chord))
            p_ctrl.append(pitch)

        # ── Cosine tip-relief ──────────────────────────────────────────────
        # Replace the hard chord drop at the tip with a smooth quarter-cosine
        # taper from c_ref (at r_relief) down to min_chord (at r_tip).
        # Only narrows the blade (min operator) so thrust is never inflated.
        if self.tip_relief_ratio > 0.0:
            r_relief   = r_tip - self.tip_relief_ratio * (r_tip - r_hub)
            c_ref      = float(np.interp(r_relief, r_ctrl, c_ctrl))
            span_relief = max(1e-9, r_tip - r_relief)
            for i, r in enumerate(r_ctrl):
                if r > r_relief:
                    t        = (r - r_relief) / span_relief
                    c_relief = (eff_min_chord
                                + (c_ref - eff_min_chord)
                                * math.cos(0.5 * math.pi * t))
                    c_ctrl[i] = min(c_ctrl[i], c_relief)

        if self.debug_splines:
            print("DEBUG chord control points (r, chord):")
            for r, c in zip(r_ctrl, c_ctrl):
                print(f"  r={r:.3f} m, c={c:.4f} m")
            r_samples = np.linspace(r_hub, r_tip, 9)
            c_spline  = CubicSpline(r_ctrl, c_ctrl)
            print("DEBUG chord samples:")
            for r in r_samples:
                print(f"  r={r:.3f} m, c={float(c_spline(r)):.4f} m")

        return CubicSpline(r_ctrl, c_ctrl), PchipInterpolator(r_ctrl, p_ctrl)

    # ─── Blades ──────────────────────────────────────────────────────────────

    @Part
    def blades(self):
        """
        Configuration Rule: instantiates n_blades Blade objects,
        each rotated evenly around the hub Z-axis.
        Rotation angle = index × (2π / n_blades).
        """
        return Sequence(
            type=Blade,
            quantify=self.n_blades,
            n_segments=self.n_segments,
            n_blades=self.n_blades,
            rpm=self.rpm,
            bemt_relaxation=self.bemt_relaxation,
            bemt_tolerance=self.bemt_tolerance,
            bemt_max_iter=self.bemt_max_iter,
            rotation_angle=lambda child: child.index
                                         * (2 * math.pi / self.n_blades),
        )

    # ─── Performance aggregation ─────────────────────────────────────────────

    @Attribute
    def total_thrust(self):
        """
        Mathematical Rule: total rotor thrust [N].
        Uses one representative blade × n_blades (all blades are
        aerodynamically identical for an axisymmetric hover rotor) to
        avoid depending on the Sequence re-quantifying when n_blades changes.
        """
        if not self.blades:
            return 0.0
        thrust = self.n_blades * self.blades[0].blade_thrust
        if thrust < 0:
            print(
                f"WARNING: Total thrust is negative ({thrust:.2f} N). "
                f"Pitch angles may be inverted or RPM too low."
            )
        return thrust

    @Attribute
    def total_torque(self):
        """
        Mathematical Rule: total rotor torque [Nm].
        Same one-blade × n_blades convention as total_thrust.
        """
        if not self.blades:
            return 0.0
        torque = self.n_blades * self.blades[0].blade_torque
        if torque < 0:
            print(
                f"WARNING: Total torque is negative ({torque:.2f} Nm). "
                f"Check spline pitch distribution."
            )
        return torque

    @Attribute
    def performance(self):
        """Mathematical Rule: thrust [N], torque [Nm] and shaft power [W]."""
        omega = self.rpm * 2 * math.pi / 60
        return {
            "thrust"      : self.total_thrust,
            "torque"      : self.total_torque,
            "shaft_power" : self.total_torque * omega,
        }

    # ─── BEMT health ─────────────────────────────────────────────────────────

    @Attribute
    def aero_health_summary(self):
        """
        Logic Rule: aggregates BEMT diagnostic flags across one
        representative blade. Returns counts plus the radii of any
        problem sections so the user can correlate against geometry.

        A section is healthy when BEMT ran, converged, and alpha was
        never clamped.  Sections at the root or tip that were skipped
        by the structural/singularity cutoffs are counted separately.
        """
        if not self.blades:
            return {
                "total_sections": 0, "healthy_count": 0,
                "stalled_radii": [], "diverged_radii": [],
                "non_converged_radii": [], "skipped_count": 0,
                "any_issue": False,
            }
        stalled, diverged, non_converged = [], [], []
        skipped = 0
        healthy = 0
        total   = 0
        for s in self.blades[0].sections:
            total += 1
            h = s.aero_health
            if h["skipped"]:
                skipped += 1
                continue
            if h["alpha_clamped"]:  stalled.append(s.radius)
            if h["diverged"]:       diverged.append(s.radius)
            if h["non_converged"]:  non_converged.append(s.radius)
            if not (h["alpha_clamped"] or h["diverged"] or h["non_converged"]):
                healthy += 1
        return {
            "total_sections"      : total,
            "healthy_count"       : healthy,
            "stalled_radii"       : stalled,
            "diverged_radii"      : diverged,
            "non_converged_radii" : non_converged,
            "skipped_count"       : skipped,
            "any_issue"           : bool(stalled or diverged or non_converged),
        }

    @Attribute
    def spanwise_distribution(self):
        """
        Diagnostic Rule: returns the clamped propeller geometry that the
        BEMT solver operates on, alongside per-section thrust and torque.
        Each row is (radius_m, chord_m, pitch_deg, dT_N, dQ_Nm).
        """
        if not self.blades:
            return []
        rows = []
        for s in self.blades[0].sections:
            aero = s.aerodynamics
            rows.append((
                s.radius,
                s.chord,
                math.degrees(s.pitch),
                aero["dT"],
                aero["dQ"],
            ))
        return rows

    # ─── Mass ────────────────────────────────────────────────────────────────

    @Attribute
    def mass(self):
        """
        Mathematical Rule: total rotor mass based on actual blade geometry.
        Computes one representative blade's volume from cross-sectional
        area × dr summed across sections, then multiplies by n_blades.
        Uses the shoelace formula on the unit-chord airfoil coordinates.
        """
        density_material = self.material_density

        # Shoelace formula: unit-chord cross-sectional area
        pts = [(p[0], p[1]) for p in self.airfoil.points]
        area_unit = 0.0
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            area_unit += x0 * y1 - x1 * y0
        area_unit = abs(area_unit) * 0.5

        if not self.blades:
            return 0.0
        single_blade_volume = sum(
            area_unit * (s.chord ** 2) * s.dr
            for s in self.blades[0].sections
        )
        return self.n_blades * single_blade_volume * density_material

    # ─── Hub geometry ────────────────────────────────────────────────────────

    @Part
    def hub(self):
        """Geometry Rule: cylindrical hub placeholder at rotor centre."""
        return Cylinder(
            radius=self.hub_radius,
            height=self.hub_height,
            centered=True,
            color="DarkSlateGray",
        )

    @Part
    def hub_flange(self):
        """
        Geometry Rule: thin flange disc below the hub to suggest a motor
        mount. Purely visual — no structural or aerodynamic function.
        """
        return Cylinder(
            radius=self.hub_radius * 1.6,
            height=self.hub_height * 0.25,
            position=translate(self.position, "z", -self.hub_height * 0.6),
            centered=True,
            color="DimGray",
        )
