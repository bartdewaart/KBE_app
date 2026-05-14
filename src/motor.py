import math

from parapy.core import Base, Input, Attribute, Part
from parapy.geom import Cylinder, translate, XOY


class ElectricMotor(Base):
    """
    Represents a candidate electric motor and evaluates whether it can
    handle the propeller's power requirements.

    Specifications are read from the motor database CSV and operating
    requirements are passed from PropulsionSystem.  Feasibility is
    checked against 80 % of rated current and power to account for
    thermal derating and modelling uncertainty.
    """

    # ─── Motor specification inputs ──────────────────────────────────────────

    #: optional input slot — motor velocity constant [RPM/V]
    kv          = Input(900)

    #: optional input slot — maximum rated power [W]
    max_power   = Input(500.0)

    #: optional input slot — maximum rated current [A]
    max_current = Input(30.0)

    #: optional input slot — winding resistance [mOhm]
    resistance  = Input(40)

    #: optional input slot — motor mass [g]
    mass        = Input(20)

    # ─── Operating-point inputs (set by PropulsionSystem) ────────────────────

    #: required input slot — required rotational speed [RPM]
    rpm_req    = Input()

    #: required input slot — required shaft torque [Nm]
    torque_req = Input()

    # ─── Geometry inputs ─────────────────────────────────────────────────────

    #: optional input slot — motor outer diameter [m]
    motor_D = Input(0.04)

    #: optional input slot — motor height [m]
    motor_h = Input(0.02)

    #: optional input slot — z-offset so motor sits below hub [m]
    motor_z_offset = Input(0.0)

    # ─── Geometry ────────────────────────────────────────────────────────────

    @Part
    def geometry(self):
        """Geometry Rule: cylindrical motor body positioned below the hub."""
        return Cylinder(
            radius=self.motor_D / 2 *3,
            height=self.motor_h,
            centered=True,
            position=translate(XOY, 'z', self.motor_z_offset),
            color="Red",
        )

    # ─── Electrical model ────────────────────────────────────────────────────

    @Attribute
    def kt(self):
        """
        Mathematical Rule: torque constant [Nm/A].
        Derived from KV rating — inverse of KV in SI units (rad/s per V).
        """
        if self.kv <= 0:
            raise ValueError(
                f"Motor KV={self.kv} is invalid. KV must be positive. "
                f"Check motor database entry."
            )
        return 1.0 / (self.kv * 2 * math.pi / 60)

    @Attribute
    def voltage_required(self):
        """Mathematical Rule: voltage needed to reach the required RPM [V]."""
        return self.rpm_req / self.kv

    @Attribute
    def current_required(self):
        """Mathematical Rule: current draw estimated from torque constant [A]."""
        return self.torque_req / self.kt

    @Attribute
    def power_required(self):
        """Mathematical Rule: total electrical power draw [W]."""
        return self.current_required * self.voltage_required

    @Attribute
    def efficiency(self):
        """
        Mathematical Rule: motor efficiency [-].
        Ratio of shaft power output to electrical power input.
        """
        shaft_power = self.torque_req * (self.rpm_req * 2 * math.pi / 60)
        if self.power_required > 0:
            eta = shaft_power / self.power_required
            if eta > 1.0:
                print(
                    f"WARNING: Motor efficiency > 100% ({eta:.2%}). "
                    f"This indicates torque_req or rpm_req may be incorrect. "
                    f"Check propeller total_torque output."
                )
            return eta
        return 0.0

    # ─── Feasibility ─────────────────────────────────────────────────────────

    @Attribute
    def feasibility_report(self):
        """
        Logic Rule: computes operating margins against rated motor limits.
        Single source of truth for the feasibility check — is_feasible
        reads its result rather than replicating the threshold arithmetic.
        Operating limits are set to 80 % of rated values to account for
        thermal effects and modelling uncertainties.
        """
        limit_current = 0.8 * self.max_current
        limit_power   = 0.8 * self.max_power
        current_ok    = self.current_required <= limit_current
        power_ok      = self.power_required   <= limit_power
        return {
            "is_feasible"    : current_ok and power_ok,
            "current_ok"     : current_ok,
            "power_ok"       : power_ok,
            "current_margin" : limit_current - self.current_required,
            "power_margin"   : limit_power   - self.power_required,
        }

    @Attribute
    def is_feasible(self):
        """
        Logic Rule: True when the motor meets both current and power limits.
        Delegates to feasibility_report to avoid duplicating threshold logic.
        """
        return self.feasibility_report["is_feasible"]
