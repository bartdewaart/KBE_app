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
    motor_db_path   = Input("data/input/motors.csv")
    diameter = Input()
    rpm = Input()

    airfoil_candidates = (["0012", "2412", "4412", "6412", "2415",
                          "4415", "23012", "23015"])
    blade_candidates = [2, 3, 4, 5, 6, 7, 8, 9 ,10]


    @Attribute
    def thrust_required(self):
        return (self.specs['MTOW'] * 9.81 / self.specs['n_rotors']) * self.specs['safety_margin']

    @Part
    def propeller(self):
        return Propeller(base_thrust=self.thrust_required,
                              diameter=self.diameter, rpm=self.rpm, thrust_to_weight = self.specs['safety_margin'])

    @Attribute
    def global_optimization(self):
        """Discrete-Continuous Hybrid Optimization Loop with Variable Normalization."""
        best_res = {"power": float('inf'), "thrust": 0.0}
        req_t = self.thrust_required  # Calculate once to save time

        for af in self.airfoil_candidates:
            for nb in self.blade_candidates:
                self.propeller.airfoil_type = af
                self.propeller.n_blades = nb

                print(f"\n{'=' * 60}")
                print(f"SEARCHING: NACA {af} | Blades: {nb} | Target Thrust: {req_t:.2f} N")
                print(f"{'-' * 60}")

                def obj(x_norm):
                    self.diameter = x_norm[0] / 10.0
                    self.rpm = x_norm[1] * 1000.0
                    perf = self.propeller.performance
                    p = perf["torque"] * (self.rpm * 2 * math.pi / 60)
                    print(
                        f"   Iter -> D: {self.diameter:.3f}m | RPM: {self.rpm:.0f} ")
                    return p

                def thrust_constraint(x_norm):
                    """Constraint: Produced Thrust >= (Base + Rotor Weight)"""
                    self.diameter = x_norm[0] / 10.0
                    self.rpm = x_norm[1] * 1000.0

                    # The propeller's internal target already includes its mass
                    total_req = self.propeller.target_thrust
                    produced = self.propeller.performance["thrust"]

                    return produced - total_req

                constraints = [
                    {'type': 'ineq', 'fun': thrust_constraint}
                ]

                x0_norm = [0.3 * 10.0, 5000 / 1000.0]
                d_max_norm = self.specs['max_diameter'] * 10.0
                bounds_norm = [(0.5, d_max_norm), (0.50, 12.0)]

                res = minimize(obj, x0_norm, method='SLSQP', bounds=bounds_norm, constraints=constraints,
                               options={'ftol': 1e-3})

                if res.success:
                    final_d = res.x[0] / 10.0
                    final_rpm = res.x[1] * 1000.0

                    # Get final thrust for the successful run
                    self.diameter, self.rpm = final_d, final_rpm
                    actual_t = self.propeller.performance["thrust"]

                    if res.fun < best_res["power"]:
                        best_res = {
                            "power": res.fun,
                            "D": final_d,
                            "RPM": final_rpm,
                            "AF": af,
                            "NB": nb,
                            "thrust": actual_t  # Store it for the report
                        }
                        print(f"*** NEW GLOBAL BEST ***")
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
        self.diameter = opt['D']
        self.rpm = opt['RPM']
        #motor = self.motor
        print(f"--- OPTIMAL UAV PROPULSION DESIGN ---\n"
              f"Airfoil: NACA {opt['AF']} | Blades: {opt['NB']}\n"
              f"Diameter: {opt['D']:.3f} m | RPM: {opt['RPM']:.0f}\n"
              f"Power: {opt['power']:.2f} W | Thrust: {opt['thrust']}")
        return opt
#