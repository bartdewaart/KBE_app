import math

from parapy.core import Base, Input, Attribute, Part
from parapy.geom import FittedCurve, Point, RotatedCurve, ScaledCurve, TranslatedCurve, Vector


class BladeSection(Base):
    """
    Represents a single radial blade station.
    Handles both BEMT aerodynamic analysis and 3D section geometry.

    NOTE: chord, pitch and other geometric properties are retrieved via
    propeller_ref (tree traversal) because they are derived from splines
    owned by Propeller and cannot be resolved as scalar Inputs at
    instantiation time without circular dependency issues.
    """

    #: optional input slot — air density [kg/m³]
    air_density = Input(1.225)

    #: required input slot — number of blades [-]
    #: propagated explicitly from Propeller via the Sequence so that
    #: ParaPy's dependency graph can track changes (tree traversal
    #: through propeller_ref is unreliable for live-mutating values).
    n_blades = Input()

    #: required input slot — rotational speed [RPM]
    #: same propagation rationale as n_blades.
    rpm = Input()

    #: optional input slot — BEMT under-relaxation factor [-]
    #: lower values are more stable but converge slower
    bemt_relaxation = Input(0.3)

    #: optional input slot — BEMT convergence tolerance [m/s]
    #: iteration stops when both v_i and v_theta change by less than this
    bemt_tolerance = Input(1e-4)

    #: optional input slot — BEMT maximum iterations [-]
    bemt_max_iter = Input(100)

    @Attribute
    def propeller_ref(self):
        """
        Configuration Rule: returns the parent Propeller object via
        tree traversal. BladeSection sits two levels below Propeller
        in the object tree (Propeller → Blade → BladeSection).
        """
        try:
            return self.parent.parent
        except AttributeError:
            raise AttributeError(
                "BladeSection could not find a Propeller at self.parent.parent. "
                "Ensure BladeSection is always instantiated inside a Blade, "
                "which itself is inside a Propeller."
            )

    @Attribute
    def radius(self):
        """Geometry Rule: radial position of this section [m]."""
        p = self.propeller_ref
        if p.n_segments < 2:
            return 0.0
        dr = (p.diameter / 2) / (p.n_segments - 1)
        return self.index * dr

    @Attribute
    def dr(self):
        """Geometry Rule: radial width of this section [m]."""
        p = self.propeller_ref
        if p.n_segments < 2:
            return p.diameter / 2
        return (p.diameter / 2) / (p.n_segments - 1)

    @Attribute
    def effective_radius(self):
        """Geometry Rule: clamp radius to the valid spline domain [hub, tip].

        The innermost section (index 0) sits at radius = 0, which lies inside
        the hub and outside the CubicSpline domain that starts at hub_radius.
        Clamping prevents the spline from extrapolating to negative or
        nonsensical chord/pitch values at the blade root.
        """
        p = self.propeller_ref
        r_tip = p.diameter / 2
        return max(p.hub_radius, min(self.radius, r_tip))

    @Attribute
    def chord(self):
        """Geometry Rule: local chord from propeller chord spline [m]."""
        c_spline, _ = self.propeller_ref.splines
        r_eval = self.effective_radius
        chord = float(c_spline(r_eval))
        if chord <= 0:
            raise ValueError(
                f"Chord at radius={self.radius:.3f}m evaluated to {chord:.4f}m. "
                f"Chord must be positive — check propeller spline control points."
            )
        return chord

    @Attribute
    def pitch(self):
        """Geometry Rule: local pitch angle from propeller pitch spline [rad]."""
        _, p_spline = self.propeller_ref.splines
        return float(p_spline(self.effective_radius))

    @Attribute
    def total_radius(self):
        """Mathematical Rule: tip radius of the full propeller [m]."""
        return self.propeller_ref.diameter / 2

    @Attribute
    def omega(self):
        """Mathematical Rule: angular velocity [rad/s]."""
        return self.rpm * 2.0 * math.pi / 60.0

    @Attribute
    def disk_area(self):
        """Mathematical Rule: total actuator disk area [m²]."""
        return math.pi * self.total_radius ** 2

    @Attribute
    def target_thrust(self):
        """Mathematical Rule: thrust requirement from parent propeller [N]."""
        return self.propeller_ref.design_thrust

    @Attribute
    def aerodynamics(self):
        """
        Mathematical Rule: iterative BEMT solver for this radial station.
        Computes local thrust (dT) and torque (dQ) contributions.
        Includes Prandtl tip-loss correction and a coupled momentum
        update on both axial (v_i) and tangential (v_theta) induced
        velocities, with relaxation for numerical stability.

        Uses chord and pitch from the propeller spline, which
        interpolates clamped control points — i.e. the realistic build
        geometry, not the unconstrained Betz design. Sections inside
        the root structural blend or outside the tip aero cutoff
        return zero contribution.

        Returns {"dT", "dQ", "health"} where health is the same dict
        exposed by self.aero_health (mirrored here so the iterative
        state set inside this method is visible to the propeller-level
        summary without recomputing).
        """
        airfoil    = self.propeller_ref.airfoil
        p = self.propeller_ref
        r_hub = p.hub_radius
        r_tip = p.diameter / 2
        root_cutoff = max(0.0, min(0.6, p.root_cutoff_ratio))
        tip_cutoff  = max(0.0, min(0.5, p.tip_cutoff_ratio))
        r_root      = r_hub + root_cutoff * (r_tip - r_hub)
        r_aero_tip  = r_tip - tip_cutoff * (r_tip - r_hub)

        if self.radius < r_root or self.radius > r_aero_tip:
            return {"dT": 0.0, "dQ": 0.0,
                    "health": {"skipped": True, "alpha_clamped": False,
                               "diverged": False, "non_converged": False}}
        v_i        = math.sqrt(
            self.target_thrust / (2.0 * self.air_density * self.disk_area)
        )
        v_theta    = 0.0
        relaxation = self.bemt_relaxation
        tolerance  = self.bemt_tolerance
        n_iter     = self.bemt_max_iter
        dT, dQ     = 0.0, 0.0
        converged = False
        alpha_clamped = False
        diverged = False

        polar = airfoil.polar_data
        alpha_min = math.radians(polar["alpha_min_deg"])
        alpha_max = math.radians(polar["alpha_max_deg"])

        for _ in range(n_iter):

            # Hover Kinematics
            v_ax  = v_i
            v_rot = self.omega * self.radius - v_theta
            v_eff = math.sqrt(v_ax ** 2 + v_rot ** 2)
            phi   = math.atan2(v_ax, v_rot)
            alpha_raw = self.pitch - phi
            if alpha_raw < alpha_min or alpha_raw > alpha_max:
                alpha_clamped = True
            alpha = max(alpha_min, min(alpha_max, alpha_raw))

            # Sectional Aerodynamic Forces
            cl, cd  = airfoil.get_cl_cd(alpha)
            l_prime = 0.5 * self.air_density * v_eff ** 2 * self.chord * cl
            d_prime = 0.5 * self.air_density * v_eff ** 2 * self.chord * cd

            # Per-blade slice contributions. n_blades enters explicitly
            # in the momentum balance below, where the annular dT_M /
            # dQ_M cover all blades.
            dT = (l_prime * math.cos(phi) - d_prime * math.sin(phi)) \
                 * self.dr
            dQ = (l_prime * math.sin(phi) + d_prime * math.cos(phi)) \
                 * self.radius * self.dr

            # Prandtl Tip-Loss Factor
            sin_phi = math.sin(phi)
            if sin_phi <= 0:
                F = 1e-6
            else:
                denom = max(1e-6, self.radius * sin_phi)
                f_tip = ((self.n_blades / 2.0)
                         * (self.total_radius - self.radius)
                         / denom)
                F = ((2.0 / math.pi)
                     * math.acos(max(0.0, min(1.0, math.exp(-f_tip)))))

            F = max(F, 1e-6)

            # Momentum Update — axial induction from thrust balance.
            # dT here is per-blade; the annular momentum thrust covers
            # all blades, so multiply by n_blades.
            v_i_new = math.sqrt(
                abs(self.n_blades * dT) / (4.0 * math.pi * self.radius
                                           * self.air_density * F * self.dr)
            )

            # Momentum Update — tangential induction from torque balance.
            # dQ_M = 4π ρ v_i v_theta F r² dr (annular, all blades).
            v_theta_denom = (4.0 * math.pi * self.air_density
                             * max(v_i_new, 1e-6) * F
                             * self.radius ** 2 * self.dr)
            v_theta_new = (self.n_blades * dQ) / v_theta_denom

            # Domain Protection: guard against divergence
            if (not math.isfinite(v_i_new) or v_i_new > 500
                    or not math.isfinite(v_theta_new)):
                if p.verbose:
                    print(
                        f"WARNING: BEMT diverged at radius={self.radius:.3f}m "
                        f"(v_i_new={v_i_new}, v_theta_new={v_theta_new}). "
                        f"Section contribution set to zero."
                    )
                diverged = True
                dT, dQ = 0.0, 0.0
                break

            # Convergence check on both induction components
            if (abs(v_i_new - v_i) < tolerance
                    and abs(v_theta_new - v_theta) < tolerance):
                converged = True
                break

            # Relaxed update
            v_i     = (1 - relaxation) * v_i     + relaxation * v_i_new
            v_theta = (1 - relaxation) * v_theta + relaxation * v_theta_new

        if alpha_clamped and p.verbose:
            print(
                f"WARNING: alpha clamped to polar range at radius="
                f"{self.radius:.3f}m. Section likely stalled or operating "
                f"outside XFOIL polar range "
                f"[{polar['alpha_min_deg']:.1f}, {polar['alpha_max_deg']:.1f}] deg."
            )

        if not converged and not diverged and p.verbose:
            print(
                f"WARNING: BEMT did not converge at radius={self.radius:.3f}m. "
                f"Last v_i={v_i:.4f} m/s, v_theta={v_theta:.4f} m/s. "
                f"Results may be inaccurate. "
                f"Consider increasing n_iter or relaxation."
            )

        return {
            "dT": dT, "dQ": dQ,
            "health": {
                "skipped"      : False,
                "alpha_clamped": alpha_clamped,
                "diverged"     : diverged,
                "non_converged": (not converged) and (not diverged),
            },
        }

    @Attribute
    def aero_health(self):
        """
        Logic Rule: per-section BEMT diagnostic flags. Lifted out of
        the aerodynamics dict so the propeller-level summary can read
        it without recomputing the iterative solver.
        """
        return self.aerodynamics["health"]

    @Attribute
    def airfoil_points(self):
        """Geometry Rule: airfoil coordinate points from parent propeller."""
        return self.propeller_ref.airfoil.points

    @Attribute
    def section_points(self):
        """Geometry Rule: map airfoil points to the local blade frame.

        Local axes:
        - X: spanwise (radial)
        - Y: tangential (chord direction)
        - Z: normal to hub plane (thickness)
        """
        return [Point(0.0, pt[0] - 0.5, pt[1]) for pt in self.airfoil_points]

    @Part(parse=False)
    def fitted_curve(self):
        return FittedCurve(points=self.section_points)

    @Part(parse=False)
    def scaled_curve(self):
        return ScaledCurve(
            curve_in=self.fitted_curve,
            reference_point=Point(0, 0, 0),
            factor=self.chord
        )

    @Part(parse=False)
    def rotated_curve(self):
        return TranslatedCurve(
            RotatedCurve(
                self.scaled_curve,
                rotation_point=Point(0, 0, 0),
                angle=self.pitch,
                vector=Vector(1, 0, 0)
            ),
            displacement=Vector(self.radius, 0, 0)
        )

    @Attribute
    def section_curve(self):
        return self.rotated_curve