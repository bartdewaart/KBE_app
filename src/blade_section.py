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
    def chord(self):
        """Geometry Rule: local chord from propeller chord spline [m]."""
        c_spline, _ = self.propeller_ref.splines
        chord = float(c_spline(self.radius))
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
        return float(p_spline(self.radius))

    @Attribute
    def total_radius(self):
        """Mathematical Rule: tip radius of the full propeller [m]."""
        return self.propeller_ref.diameter / 2

    @Attribute
    def n_blades(self):
        """Configuration Rule: number of blades from parent propeller."""
        return self.propeller_ref.n_blades

    @Attribute
    def rpm(self):
        """Configuration Rule: rotational speed from parent propeller [RPM]."""
        return self.propeller_ref.rpm

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
        Includes Prandtl tip-loss correction and momentum update
        with relaxation for numerical stability.
        """
        airfoil    = self.propeller_ref.airfoil
        p = self.propeller_ref
        r_hub = p.hub_radius
        r_tip = p.diameter / 2
        root_cutoff = max(0.0, min(0.6, p.root_cutoff_ratio))
        r_root = r_hub + root_cutoff * (r_tip - r_hub)
        if self.radius < r_root:
            return {"dT": 0.0, "dQ": 0.0}
        v_i        = math.sqrt(
            self.target_thrust / (2.0 * self.air_density * self.disk_area)
        )
        v_theta    = 0.0
        relaxation = 0.3  # was 0.1 — more aggressive relaxation
        tolerance  = 1e-4  # was 1e-5 — slightly looser tolerance
        n_iter     = 100  # 500 is overkill if it's diverging anyway, put at 100 for testing now
        dT, dQ     = 0.0, 0.0
        v_theta_new = 0.0
        converged = False
        for _ in range(n_iter):

            # Hover Kinematics
            v_ax  = v_i
            v_rot = self.omega * self.radius - v_theta
            v_eff = math.sqrt(v_ax ** 2 + v_rot ** 2)
            phi   = math.atan2(v_ax, v_rot)
            alpha = self.pitch - phi

            polar = airfoil.polar_data
            alpha_min = math.radians(polar["alpha_min_deg"])
            alpha_max = math.radians(polar["alpha_max_deg"])
            alpha = max(alpha_min, min(alpha_max, alpha))

            # Sectional Aerodynamic Forces
            cl, cd  = airfoil.get_cl_cd(alpha)
            l_prime = 0.5 * self.air_density * v_eff ** 2 * self.chord * cl
            d_prime = 0.5 * self.air_density * v_eff ** 2 * self.chord * cd

            dT = (l_prime * math.cos(phi) - d_prime * math.sin(phi)) \
                 * self.n_blades * self.dr
            dQ = (l_prime * math.sin(phi) + d_prime * math.cos(phi)) \
                 * self.radius * self.n_blades * self.dr

            # Prandtl Tip-Loss Factor
            # Domain Protection: guard against sin(phi) <= 0
            sin_phi = math.sin(phi)
            if sin_phi <= 0:
                F = 1e-6
            else:
                f_tip = ((self.n_blades / 2.0)
                         * (self.total_radius - self.radius)
                         / (self.radius * sin_phi))
                F = ((2.0 / math.pi)
                     * math.acos(max(0.0, min(1.0, math.exp(-f_tip)))))

            F = max(F, 1e-6)

            # Momentum Update
            v_i_new = math.sqrt(
                abs(dT) / (4.0 * math.pi * self.radius
                           * self.air_density * F * self.dr)
            )

            # Domain Protection: guard against divergence
            if not math.isfinite(v_i_new) or v_i_new > 500:
                # Unphysical result — clamp and break
                dT, dQ = 0.0, 0.0
                break

            # Convergence check
            if abs(v_i_new - v_i) < tolerance:
                converged = True
                break

            # Relaxed update
            v_i     = (1 - relaxation) * v_i     + relaxation * v_i_new
            v_theta = (1 - relaxation) * v_theta + relaxation * v_theta_new

        if not converged:
            print(
                f"WARNING: BEMT did not converge at radius={self.radius:.3f}m. "
                f"Last v_i={v_i:.4f} m/s. Results may be inaccurate. "
                f"Consider increasing n_iter or relaxation."
            )

        return {"dT": dT, "dQ": dQ}

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

    # Each func below receives parse=False because they depend on runtime-computed chord, radius, pitch
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