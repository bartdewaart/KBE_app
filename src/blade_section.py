import math
from parapy.core import Base, Input, Attribute, Part
from parapy.geom import FittedCurve, ScaledShape, RotatedShape, TranslatedShape, Vector
from .airfoil import Airfoil

class BladeSection(Base):
    air_density = Input(1.225)

    @Attribute
    def propeller_ref(self):
        curr = self.parent
        while curr is not None:
            if hasattr(curr, "n_segments"):
                return curr
            curr = getattr(curr, "parent", None)
        raise AttributeError("BladeSection could not find a Propeller in its parent chain.")

    @Part
    def airfoil_geom(self):
        return Airfoil(naca_code=self.propeller_ref.airfoil_type, reynolds=300000, chord = self.chord)

    @Part
    def section_curve(self):
        return TranslatedShape(RotatedShape(ScaledShape(
            FittedCurve(points=self.airfoil_geom.points),
            scale_factor=self.chord),
            angle=self.pitch,
            vector=Vector(0,1,0)),
            displacement=Vector(0, 0, self.radius))

    @Attribute
    def radius(self):
        p = self.propeller_ref
        dr = (p.diameter / 2 - 0.02) / p.n_segments
        return 0.02 + (self.index + 0.5) * dr

    @Attribute
    def chord(self):
        c_spline, _ = self.propeller_ref.splines
        return float(c_spline(self.radius))

    @Attribute
    def pitch(self):
        _, p_spline = self.propeller_ref.splines
        return float(p_spline(self.radius))

    @Attribute
    def dr(self):
        p = self.propeller_ref
        # Calculation: (Tip Radius - Hub Radius) / Number of segments
        return (p.diameter / 2 - 0.02) / p.n_segments

    @Attribute
    def total_radius(self):
        return self.propeller_ref.diameter / 2

    @Attribute
    def n_blades(self):
        return self.propeller_ref.n_blades

    @Attribute
    def rpm(self):
        return self.propeller_ref.rpm

    @Attribute
    def airfoil_obj(self):
        # This points to the airfoil defined in the Propeller class
        return self.propeller_ref.airfoil

    @Attribute
    def aerodynamics(self):
        p = self.propeller_ref  # Shortcut

        omega = p.rpm * 2.0 * math.pi / 60.0
        v_rot = omega * self.radius

        # Use the diameter and target_thrust from the PROPELLER level
        r_tip = p.diameter / 2
        v_ax = math.sqrt(p.target_thrust / (2.0 * self.air_density * math.pi * r_tip ** 2))

        phi = math.atan2(v_ax, v_rot)
        v_eff = math.sqrt(v_ax ** 2 + v_rot ** 2)

        # Get airfoil data from the propeller's airfoil part
        cl, cd = p.airfoil.get_cl_cd(self.pitch - phi)
        re = (self.air_density * v_eff * self.chord) / 1.8e-5

        l_prime = 0.5 * self.air_density * v_eff ** 2 * self.chord * cl
        d_prime = 0.5 * self.air_density * v_eff ** 2 * self.chord * cd

        # Calculate for ALL blades
        dT = (l_prime * math.cos(phi) - d_prime * math.sin(phi)) * p.n_blades * self.dr
        dQ = (l_prime * math.sin(phi) + d_prime * math.cos(phi)) * self.radius * p.n_blades * self.dr
        return {"dT": dT, "dQ": dQ}

