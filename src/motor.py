import math

from parapy.core import Base, Input, Attribute
from .config import get_value


class ElectricMotor(Base):
    """
    Represents a candidate electric motor and evaluates
    whether it can handle the propeller's power requirements.
    Motor specifications are read from the motor database CSV
    and operating requirements are passed from PropulsionSystem.
    """

    #: optional input slot — motor velocity constant [RPM/V]
    kv          = Input(get_value("motor", "kv", default=900))

    #: optional input slot — maximum rated power [W]
    max_power   = Input(get_value("motor", "max_power", default=500.0))

    #: optional input slot — maximum rated current [A]
    max_current = Input(get_value("motor", "max_current", default=30.0))

    #: optional input slot — winding resistance [mOhm]
    resistance  = Input(get_value("motor", "resistance", default=40))

    #: optional input slot — motor mass [g]
    mass        = Input(get_value("motor", "mass", default=20))

    #: required input slot — required rotational speed [RPM]
    rpm_req     = Input()

    #: required input slot — required shaft torque [Nm]
    torque_req  = Input()

    @Attribute
    def kt(self):
        """
        Mathematical Rule: torque constant [Nm/A].
        Derived from KV rating — inverse of KV in SI units.
        """
        if self.kv <= 0:
            raise ValueError(
                f"Motor KV={self.kv} is invalid. KV must be positive. "
                f"Check motor database entry."
            )
        return 1.0 / (self.kv * 2 * math.pi / 60)

    @Attribute
    def voltage_required(self):
        """
        Mathematical Rule: voltage needed to reach the required RPM [V].
        """
        return self.rpm_req / self.kv

    @Attribute
    def current_required(self):
        """
        Mathematical Rule: current draw estimated from torque constant [A].
        """
        return self.torque_req / self.kt

    @Attribute
    def power_required(self):
        """
        Mathematical Rule: total electrical power draw [W].
        """
        r_ohm = self.resistance / 1000.0
        return self.current_required * self.voltage_required + (self.current_required ** 2) * r_ohm

    @Attribute
    def efficiency(self):
        """
        Mathematical Rule: motor efficiency [-].
        Ratio of shaft power output to electrical power input.
        """
        shaft_power = self.torque_req * (self.rpm_req * 2 * math.pi / 60)
        if self.power_required > 0:
            eta = shaft_power / self.power_required
            if eta > 1.05:
                print(
                    f"WARNING: Motor efficiency > 100% ({eta:.2%}). "
                    f"This indicates torque_req or rpm_req may be incorrect. "
                    f"Check propeller total_torque output."
                )
            return eta
        return 0.0

    @Attribute
    def is_feasible(self):
        """
        Logic Rule: checks feasibility against rated motor limits.
        Operating margins set to 80% of rated current and power
        to account for thermal effects and modelling uncertainties.
        """
        current_ok = self.current_required <= 0.8 * self.max_current
        power_ok   = self.power_required   <= 0.8 * self.max_power
        return current_ok and power_ok

    @Attribute
    def feasibility_report(self):
        """
        Logic Rule: returns a detailed feasibility summary showing
        which constraints pass or fail and by what margin.
        References is_feasible to avoid duplicating logic.
        """
        return {
            "is_feasible"    : self.is_feasible,
            "current_ok"     : self.current_required <= 0.8 * self.max_current,
            "power_ok"       : self.power_required   <= 0.8 * self.max_power,
            "current_margin" : 0.8 * self.max_current - self.current_required,
            "power_margin"   : 0.8 * self.max_power   - self.power_required,
        }