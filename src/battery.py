from parapy.core import Base, Input, Attribute
from .config import get_value

class Battery(Base):
    """
    Represents a battery with energy density, capacity, and weight.
    """
    energy_density = Input(get_value("battery", "energy_density", default=200))  # Wh/kg, default value
    capacity = Input(get_value("battery", "capacity", default=1000))  # Wh, default value

    @Attribute
    def weight(self):
        """
        Calculates the battery weight based on capacity and energy density.
        """
        return self.capacity / self.energy_density

    @Attribute
    def endurance(self):
        """
        Calculates the endurance (hours) based on capacity and power consumption.
        Assumes power consumption is provided by the parent system.
        """
        if hasattr(self.parent, 'power_consumption'):
            return self.capacity / self.parent.power_consumption
        return None
