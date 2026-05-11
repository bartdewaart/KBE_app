import math

from parapy.core import Base, Input, Attribute, Part
from parapy.geom import FittedCurve, ScaledShape, RotatedShape, TranslatedShape, Vector


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
        p  = self.propeller_ref
        dr = (p.diameter / 2 - 0.02) / p.n_segments
        return 0.02 + (self.index + 0.5) * dr

    @Attribute
    def dr(self):
        """Geometry Rule: radial width of this section [m]."""
        p = self.propeller_ref
        return (p.diameter / 2 - 0.02) / p.n_segments

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
        v_i        = math.sqrt(
            self.target_thrust / (2.0 * self.air_density * self.disk_area)
        )
        v_theta    = 0.0
        relaxation = 0.1
        tolerance  = 1e-5
        n_iter     = 500
        dT, dQ     = 0.0, 0.0

        converged = False
        for _ in range(n_iter):

            # Hover Kinematics
            v_ax  = v_i
            v_rot = self.omega * self.radius - v_theta
            v_eff = math.sqrt(v_ax ** 2 + v_rot ** 2)
            phi   = math.atan2(v_ax, v_rot)
            alpha = self.pitch - phi

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

            # Domain Protection: guard against division by zero in swirl
            if v_i_new > 0:
                v_theta_new = (dQ / (4.0 * math.pi * self.radius ** 2
                               * self.air_density * v_i_new * F * self.dr))
            else:
                v_theta_new = 0.0

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

    @Part
    def section_curve(self):
        """
        Geometry Rule: generates the 3D cross-section curve at this
        radial station by scaling, rotating and translating the airfoil.
        """
        return TranslatedShape(
            RotatedShape(
                ScaledShape(
                    FittedCurve(points=self.propeller_ref.airfoil.points),
                    scale_factor=self.chord
                ),
                angle=self.pitch,
                vector=Vector(0, 1, 0)
            ),
            displacement=Vector(0, 0, self.radius)
        )