import math
from parapy.core import Base, Input, Attribute, Part, Sequence
from src.airfoil import Airfoil
from src.blade import Blade


class BasePropeller(Base):
    """Abstract base class for all propeller types."""

    @Attribute
    def total_thrust(self):
        raise NotImplementedError("Subclasses must implement total_thrust")

    @Attribute
    def total_torque(self):
        raise NotImplementedError("Subclasses must implement total_torque")


class HoverPropeller(BasePropeller):
    """Concrete implementation for static hover using BEMT."""

    diameter      = Input(0.4)
    hub_diameter  = Input(0.04)
    n_blades      = Input(2)
    rpm           = Input(5000)
    n_segments    = Input(15)
    chord         = Input(0.03)
    pitch         = Input(math.radians(15))
    target_thrust = Input()

    @Part
    def airfoil(self):
        return Airfoil(naca_code="4412", reynolds=300000)

    @Part
    def blades(self):
        return Sequence(
            type=Blade,
            quantify=self.n_blades,
            pass_down="n_segments, chord, pitch, rpm, n_blades, target_thrust, airfoil",
            total_radius=self.diameter / 2,
            hub_radius=self.hub_diameter / 2,
        )

    @Attribute
    def total_thrust(self):
        return sum(blade.blade_thrust for blade in self.blades)

    @Attribute
    def total_torque(self):
        return sum(blade.blade_torque for blade in self.blades)