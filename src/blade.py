import math
from parapy.core import Base, Input, Attribute, Part, Sequence
from src.blade_section import BladeSection


class Blade(Base):
    """One physical blade, owns the spanwise BladeSection sequence"""

    n_segments    = Input(15)
    chord         = Input(0.03)
    pitch         = Input(math.radians(15))
    total_radius  = Input()
    hub_radius    = Input()
    n_blades      = Input()
    rpm           = Input()
    air_density   = Input(1.225)
    target_thrust = Input()
    airfoil       = Input()

    @Attribute
    def dr(self):
        return (self.total_radius - self.hub_radius) / self.n_segments

    @Part
    def sections(self):
        return Sequence(
            type=BladeSection,
            quantify=self.n_segments,
            pass_down="chord, pitch, n_blades, rpm, air_density, target_thrust, airfoil, total_radius",
            radius=lambda item: self.hub_radius + (item.index + 0.5) * self.dr,
            dr=self.dr,
        )

    @Attribute
    def blade_thrust(self):
        return sum(s.aerodynamics["dT"] for s in self.sections)

    @Attribute
    def blade_torque(self):
        return sum(s.aerodynamics["dQ"] for s in self.sections)

    @Attribute
    def chord_distribution(self):
        return [s.chord for s in self.sections]

    @Attribute
    def twist_distribution(self):
        return [math.degrees(s.pitch) for s in self.sections]