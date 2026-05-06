import csv

from OCC.wrapper.AIS import AIS_Trihedron
from parapy.core import Base, Input, Attribute, Part, Sequence
from .propeller import Propeller
from .motor import ElectricMotor
from scipy.optimize import minimize
import math


class PropulsionSystem(Base):
    """
    Root class, couples the propeller design with motor selection.
    Reads motor database and finds the best feasible match.
    """

    # Mission inputs
    specs = Input()  # From Excel
    #motors = Input()  # From Excel

    motor_db_path   = Input("data/input/motors.csv")

    airfoil_candidates = (["0012", "2412", "4412"])
                              #, "6412", "2415",)
                          #"4415", "23012", "23015"]
    blade_candidates = [2, 3, 4, 5, 6, 7, 8]


    @Attribute
    def thrust_required(self):
        return (self.specs['MTOW'] * 9.81 / self.specs['n_rotors']) * self.specs['safety_margin']

    @Part
    def propeller(self):
        return Propeller(target_thrust=self.thrust_required,
                              diameter=0.3, rpm=5000)

    @Attribute
    def global_optimization(self):
        """Discrete-Continuous Hybrid Optimization Loop."""
        best_res = {"power": float('inf')}

        for af in self.airfoil_candidates:
            for nb in self.blade_candidates:
                self.propeller.airfoil_type = af
                self.propeller.n_blades = nb

                def obj(x):
                    self.propeller.diameter, self.propeller.rpm = x[0], x[1]
                    return self.propeller.performance["torque"] * (x[1] * 2 * math.pi / 60)

                res = minimize(obj, [0.3, 5000], bounds=[(0.1, self.specs['max_diameter']), (2000, 8000)])
                if res.fun < best_res["power"]:
                    best_res = {"power": res.fun, "D": res.x[0], "RPM": res.x[1], "AF": af, "NB": nb}
        return best_res


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
                torque_req   = self.propeller.total_torque / self.specs['n_rotors'],
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

    @Part(parse=False)
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
            torque_req  = self.propeller.total_torque / self.specs['n_rotors'],
            label       = best_name,
        )

    @Attribute
    def generate_report(self):
        opt = self.global_optimization
        #motor = self.motor
        print(f"--- OPTIMAL UAV PROPULSION DESIGN ---\n"
              f"Airfoil: NACA {opt['AF']} | Blades: {opt['NB']}\n"
              f"Diameter: {opt['D']:.3f} m | RPM: {opt['RPM']:.0f}\n"
              f"Power: {opt['power']:.2f} W ")
        return opt
#