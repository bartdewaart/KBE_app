import math
from parapy.core import Base, Input, Attribute, Part
from parapy.geom import Position, translate, rotate
from .config import get_value


class BladeSection(Base):
    position = Input(Position())
    thickness_factor = Input(get_value("blade_section", "thickness_factor", default=1.0))
    pitch_deg_max = Input(get_value("blade_section", "pitch_deg_max", default=85.0))
    @Attribute
    def propeller_ref(self):
        """Climbs the tree to find inputs on the Propeller."""
        curr = self.parent
        while curr is not None:
            if hasattr(curr, "base_thrust"): return curr
            curr = getattr(curr, "parent", None)
        return None



    @Attribute
    def airfoil_class(self):
        """Move the local import to an attribute."""
        from .airfoil import Airfoil
        return Airfoil

    @Part
    def airfoil(self):
        return self.airfoil_class(
            naca_code=self.propeller_ref.airfoil_type,
            chord=self.chord,
            thickness_factor=self.thickness_factor,
            position=translate(
                rotate(self.position, "y", self.pitch_deg),
                "z", self.radius
                )
            )

    @Attribute
    def positioned_curve(self):
        return self.airfoil.curve

    @Attribute
    def radius(self):
        return self.propeller_ref.hub_radius + (self.index + 0.5) * self.dr

    @Attribute
    def dr(self):
        return (self.propeller_ref.diameter / 2 - self.propeller_ref.hub_radius) / self.propeller_ref.n_segments

    @Attribute
    def chord(self):
        value = float(self.propeller_ref.splines[0](self.radius))
        min_chord = getattr(self.propeller_ref, "min_chord", 0.005)
        if not math.isfinite(value):
            return min_chord
        return max(min_chord, abs(value))

    @Attribute
    def pitch_rad(self):
        return float(self.propeller_ref.splines[1](self.radius))

    @Attribute
    def pitch_deg(self):
        pitch_deg = math.degrees(self.pitch_rad)
        limit = max(1.0, float(self.pitch_deg_max))
        return max(-limit, min(limit, pitch_deg))

    @Attribute
    def aerodynamics(self):
        p = self.propeller_ref
        v_ax = math.sqrt(p.design_thrust / (2.0 * 1.225 * math.pi * (p.diameter / 2) ** 2))
        v_rot = (p.rpm * math.pi / 30) * self.radius
        phi = math.atan2(v_ax, v_rot)
        v_eff = math.sqrt(v_ax ** 2 + v_rot ** 2)
        cl, cd = self.airfoil.get_cl_cd(self.pitch_rad - phi)
        dT = (0.5 * 1.225 * v_eff ** 2 * self.chord * cl * math.cos(phi)) * p.n_blades * self.dr
        dQ = (0.5 * 1.225 * v_eff ** 2 * self.chord * cl * math.sin(phi)) * self.radius * p.n_blades * self.dr
        return {"dT": dT, "dQ": dQ}