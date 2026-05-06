import csv
from parapy.core import Base, Input, Attribute, Part, Sequence
from src.propeller import HoverPropeller
from src.motor import ElectricMotor


class PropulsionSystem(Base):
    """
    Root class, couples the propeller design with motor selection.
    Reads motor database and finds the best feasible match.
    """

    # Mission inputs
    mtow            = Input(2.0)    # kg
    n_rotors        = Input(4)
    safety_margin   = Input(1.5)
    motor_db_path   = Input("data/input/motors.csv")

    @Attribute
    def thrust_required(self):
        """Design thrust per rotor including safety margin"""
        return (self.mtow * 9.81 / self.n_rotors) * self.safety_margin

    @Part
    def propeller(self):
        return HoverPropeller(
            target_thrust=self.thrust_required
        )

    @Attribute
    def motor_database(self):
        """
        Integration Rule: reads motors.csv and returns
        a list of dicts, one per motor
        """
        motors = []
        with open(self.motor_db_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                motors.append({
                    "name"        : row["name"],
                    "kv"          : float(row["kv"]),
                    "max_power"   : float(row["max_power_w"]),
                    "max_current" : float(row["max_current_a"]),
                    "resistance"  : float(row["resistance_mohm"]),
                    "mass"        : float(row["mass_g"]),
                })
        return motors

    @Attribute
    def candidate_motors(self):
        """
        Creates an ElectricMotor object for every motor in the
        database and evaluates feasibility against the propeller
        """
        candidates = []
        for m in self.motor_database:
            motor = ElectricMotor(
                kv           = m["kv"],
                max_power    = m["max_power"],
                max_current  = m["max_current"],
                resistance   = m["resistance"],
                mass         = m["mass"],
                rpm_req      = self.propeller.rpm,
                torque_req   = self.propeller.total_torque / self.n_rotors,
            )
            candidates.append((m["name"], motor))
        return candidates

    @Attribute
    def feasible_motors(self):
        """Filters candidate motors to only feasible ones"""
        return [(name, motor) for name, motor in self.candidate_motors
                if motor.is_feasible]

    @Attribute
    def best_motor(self):
        """
        Ranks feasible motors by efficiency and returns the best one.
        Returns None if no feasible motor is found
        """
        if not self.feasible_motors:
            print("WARNING: No feasible motor found. Consider adjusting propeller design.")
            return None
        return max(self.feasible_motors, key=lambda x: x[1].efficiency)

    @Part
    def motor(self):
        """
        Instantiates the best motor as a proper ParaPy Part
        so it appears in the model tree
        """
        best_name, best = self.best_motor
        return ElectricMotor(
            kv          = best.kv,
            max_power   = best.max_power,
            max_current = best.max_current,
            resistance  = best.resistance,
            mass        = best.mass,
            rpm_req     = self.propeller.rpm,
            torque_req  = self.propeller.total_torque / self.n_rotors,
            label       = best_name,
        )