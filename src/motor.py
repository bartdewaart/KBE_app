import math
from parapy.core import Base, Input, Attribute


class ElectricMotor(Base):
    """
    Represents a candidate electric motor and evaluates
    whether it can handle the propeller's power requirements.
    """

    # Motor specifications (from database)
    kv            = Input(900)      # RPM/V
    max_power     = Input(500.0)    # W
    max_current   = Input(30.0)     # A
    resistance    = Input(40)       # mOhm
    mass          = Input(20)       # g

    # Operating requirements (passed from PropulsionSystem)
    rpm_req       = Input()         # required RPM
    torque_req    = Input()         # required torque in Nm

    @Attribute
    def kt(self):
        """Torque constant, inverse of KV in SI units"""
        return 1.0 / (self.kv * 2 * math.pi / 60)

    @Attribute
    def voltage_required(self):
        """Voltage needed to reach the required RPM"""
        return self.rpm_req / self.kv

    @Attribute
    def current_required(self):
        """Current draw estimated from torque constant"""
        return self.torque_req / self.kt

    @Attribute
    def power_required(self):
        """Electrical power draw"""
        return self.current_required * self.voltage_required

    @Attribute
    def efficiency(self):
        """Motor efficiency, shaft power out / electrical power in"""
        shaft_power = self.torque_req * (self.rpm_req * 2 * math.pi / 60)
        if self.power_required > 0:
            return shaft_power / self.power_required
        return 0.0

    @Attribute
    def is_feasible(self):
        """
        Logic Rule: checks all three feasibility constraints.
        Operating at max 80% of rated limits for safety margin
        """
        current_ok = self.current_required <= 0.8 * self.max_current
        power_ok   = self.power_required   <= 0.8 * self.max_power
        return current_ok and power_ok

    @Attribute
    def feasibility_report(self):
        """Returns a readable summary of which constraints pass or fail"""
        return {
            "current_ok" : self.current_required <= 0.8 * self.max_current,
            "power_ok"   : self.power_required   <= 0.8 * self.max_power,
            "is_feasible": self.is_feasible,
            "current_margin": 0.8 * self.max_current - self.current_required,
            "power_margin"  : 0.8 * self.max_power   - self.power_required,
        }