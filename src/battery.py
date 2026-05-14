from parapy.core import Base, Input, Attribute


class Battery(Base):
    """
    Energy-storage model for the propulsion system.

    Mass is sized from the electrical power draw and the user's
    endurance target — the battery is exactly large enough to fly
    for target_endurance_min minutes at the optimum hover power.
    Capacity and endurance are reported back as derived quantities.
    """

    #: required input slot — total electrical power draw across all rotors [W]
    power_electrical = Input()

    #: required input slot — desired hover endurance [min]
    target_endurance_min = Input()

    #: optional input slot — pack-level gravimetric energy density [Wh/kg]
    #: 2025 SOTA Li-ion / pouch cell: 250–300 Wh/kg
    energy_density = Input(250.0)

    #: optional input slot — pack-to-shaft conversion efficiency [-]
    efficiency = Input(0.85)

    @Attribute
    def mass(self):
        """Battery mass sized for the requested endurance [kg]."""
        energy_required_Wh = self.power_electrical * (self.target_endurance_min / 60.0)
        return energy_required_Wh / (self.energy_density * self.efficiency)

    @Attribute
    def capacity_Wh(self):
        """Battery nameplate capacity [Wh]."""
        return self.mass * self.energy_density

    @Attribute
    def endurance_min(self):
        """Reported endurance for the sized battery [min]."""
        if self.power_electrical <= 0:
            return 0.0
        return 60.0 * (self.capacity_Wh * self.efficiency) / self.power_electrical
