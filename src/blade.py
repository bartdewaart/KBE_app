import math

from parapy.core import Base, Input, Attribute, Part, Sequence
from parapy.geom import LoftedSurface, RotatedShape, Vector, Point

from src.blade_section import BladeSection


class Blade(Base):
    """
    Represents a single physical blade.
    Owns the spanwise sequence of BladeSection objects and lofts
    them into a 3D surface. Also aggregates per-blade aerodynamic
    performance from its sections.

    Each blade instance is rotated by rotation_angle around the
    hub axis (Z-axis) to produce the full rotor assembly.
    rotation_angle is computed and passed by Propeller as:
    index * (2π / n_blades)
    """

    #: required input slot — rotation angle around hub Z-axis [rad]
    #: computed and passed by Propeller: index * (2π / n_blades)
    rotation_angle = Input()

    #: optional input slot — number of spanwise analysis sections
    n_segments = Input(30)

    #: required input slot — number of blades [-] (forwarded to sections
    #: so ParaPy tracks changes through the Sequence chain).
    n_blades = Input()

    #: required input slot — rotational speed [RPM] (forwarded to sections).
    rpm = Input()

    #: optional input slot — BEMT under-relaxation factor (forwarded to sections)
    bemt_relaxation = Input(0.3)

    #: optional input slot — BEMT convergence tolerance [m/s] (forwarded to sections)
    bemt_tolerance = Input(1e-4)

    #: optional input slot — BEMT maximum iterations (forwarded to sections)
    bemt_max_iter = Input(100)

    @Part
    def sections(self):
        """
        Configuration Rule: instantiates n_segments BladeSection
        objects spanning from hub to tip. Chord, pitch and radius
        are computed internally by each section via propeller_ref.
        n_blades and rpm are propagated as Inputs so ParaPy's
        dependency graph correctly invalidates BEMT on changes.
        """
        return Sequence(
            type=BladeSection,
            quantify=self.n_segments,
            n_blades=self.n_blades,
            rpm=self.rpm,
            bemt_relaxation=self.bemt_relaxation,
            bemt_tolerance=self.bemt_tolerance,
            bemt_max_iter=self.bemt_max_iter,
        )

    @Attribute
    def blade_thrust(self):
        """
        Mathematical Rule: total thrust contribution of this blade [N].
        Aggregates dT from all spanwise sections.
        """
        thrust = sum(s.aerodynamics["dT"] for s in self.sections)
        if thrust < 0:
            print(
                f"WARNING: Blade thrust is negative ({thrust:.3f} N). "
                f"This usually means pitch angles are inverted or RPM is too low. "
                f"Check propeller spline pitch distribution."
            )
        return thrust

    @Attribute
    def blade_torque(self):
        """
        Mathematical Rule: total torque contribution of this blade [Nm].
        Aggregates dQ from all spanwise sections.
        """
        torque = sum(s.aerodynamics["dQ"] for s in self.sections)
        if torque < 0:
            print(
                f"WARNING: Blade torque is negative ({torque:.3f} Nm). "
                f"This usually means pitch angles are inverted or RPM is too low. "
                f"Check propeller spline pitch distribution."
            )
        return torque

    @Attribute
    def rotation_angle_value(self):
        value = self.rotation_angle
        if callable(value):
            value = value(self)
        return float(value)

    @Part
    def surface(self):
        """
        Geometry Rule: lofts the section curves into the 3D blade skin.
        Hidden because rotated_surface is the rendered copy in the rotor
        assembly; this intermediate node exists only for LoftedSurface.
        """
        return LoftedSurface(
            profiles=[s.section_curve for s in self.sections],
            hidden=True,
        )

    @Attribute
    def health_color(self):
        """
        Visual cue: green when the rotor's BEMT ran clean for this
        design, amber if any section stalled, red if any diverged.
        Reads parent.aero_health_summary so the colour reacts whenever
        the design changes.
        """
        try:
            health = self.parent.aero_health_summary
        except AttributeError:
            return "ForestGreen"
        if health["diverged_radii"] or health["non_converged_radii"]:
            return "Crimson"
        if health["stalled_radii"]:
            return "Goldenrod"
        return "ForestGreen"

    @Part
    def rotated_surface(self):
        """
        Geometry Rule: rotates the lofted blade surface by rotation_angle
        around the Z-axis (hub axis) to position this blade in the
        full rotor assembly. Coloured by aero health.
        """
        return RotatedShape(
            shape_in=self.surface,
            rotation_point=Point(0, 0, 0),
            vector=Vector(0, 0, 1),
            angle=self.rotation_angle_value,
            color=self.health_color,
        )