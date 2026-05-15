from parapy.core import Base, Input, Attribute


class Battery(Base):
    """
    Energy-storage model for the propulsion system.

    Capacity (Wh) is set externally by PropulsionSystem once the optimizer
    has fixed the battery energy budget.  Mass derives from capacity and
    energy density.  Endurance is reported given the current electrical
    power draw.
    """

    #: required input slot — fixed energy budget for this battery [Wh]
    capacity_Wh = Input()

    #: required input slot — total electrical power draw across all rotors [W]
    power_electrical = Input()

    #: optional input slot — pack-level gravimetric energy density [Wh/kg]
    energy_density = Input(250.0)

    #: optional input slot — pack-to-shaft conversion efficiency [-]
    efficiency = Input(0.85)

    @Attribute
    def mass(self):
        """Battery mass from fixed energy budget [kg]."""
        return self.capacity_Wh / max(self.energy_density, 1e-6)

    @Attribute
    def endurance_min(self):
        """Achievable hover endurance at the current power draw [min]."""
        if self.power_electrical <= 0:
            return 0.0
        return 60.0 * self.capacity_Wh * self.efficiency / self.power_electrical
