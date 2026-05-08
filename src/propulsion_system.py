import csv
import math
import os

from parapy.core import Base, Input, Attribute, Part
from scipy.optimize import minimize

from .motor import ElectricMotor
from .propeller import Propeller


class PropulsionSystem(Base):
    """
    Root class — couples the propeller design with motor selection.
    Reads mission specifications from Excel input, runs a hybrid
    discrete-continuous optimization to find the optimal propeller
    design, reads the motor database and selects the best feasible
    motor match.
    """

    #: required input slot — mission specifications dict from Excel
    #: expected keys: MTOW, n_rotors, safety_margin, max_diameter
    specs = Input()

    #: optional input slot — path to motor database CSV file
    motor_db_path = Input("data/input/motors.csv")

    #: optional input slot — initial/current propeller diameter [m]
    #: updated by optimization to optimal value
    diameter = Input(0.3)

    #: optional input slot — initial/current rotational speed [RPM]
    #: updated by optimization to optimal value
    rpm = Input(5000.0)

    #: optional input slot — NACA airfoil candidates to search over
    airfoil_candidates = Input(["0012", "2412", "4412", "6412",
                                "2415", "4415", "23012", "23015"])

    #: optional input slot — blade count candidates to search over
    blade_candidates = Input([2, 3, 4, 5, 6, 7, 8, 9, 10])

    @Attribute
    def thrust_required(self):
        """
        Mathematical Rule: required thrust per rotor [N].
        Derived from MTOW, number of rotors and safety margin.
        """
        return ((self.specs['MTOW'] * 9.81 / self.specs['n_rotors'])
                * self.specs['safety_margin'])

    @Part
    def propeller(self):
        """
        Configuration Rule: instantiates the Propeller object with
        mission-derived thrust requirement and current diameter/RPM.
        """
        return Propeller(
            base_thrust   = self.thrust_required,
            diameter      = self.diameter,
            rpm           = self.rpm,
            safety_margin = self.specs['safety_margin']
        )


    def run_optimization(self):
        """
        Integration Rule: Discrete-Continuous Hybrid Optimization Loop.
        Searches over all airfoil and blade count candidates (discrete),
        and uses scipy SLSQP to optimize diameter and RPM (continuous).
        Minimizes shaft power subject to thrust constraint.

        Implemented as a regular method (not @Attribute) because it
        intentionally mutates Input slots during the search loop —
        a necessary deviation from functional style when coupling
        ParaPy models to external optimizers.

        Call explicitly from main.py after instantiation:
            result = system.run_optimization()
            display(system)
        """
        best_res = {"power": float('inf'), "thrust": 0.0}
        req_t = self.thrust_required

        for af in self.airfoil_candidates:
            for nb in self.blade_candidates:
                self.propeller.airfoil_type = af
                self.propeller.n_blades = nb

                print(f"\n{'=' * 60}")
                print(f"SEARCHING: NACA {af} | Blades: {nb} "
                      f"| Target Thrust: {req_t:.2f} N")
                print(f"{'-' * 60}")

                def obj(x_norm):
                    """Objective: minimise shaft power."""
                    self.diameter = x_norm[0] / 10.0
                    self.rpm = x_norm[1] * 1000.0
                    p = self.propeller.performance["shaft_power"]
                    print(f"   Iter -> D: {self.diameter:.3f}m "
                          f"| RPM: {self.rpm:.0f}")
                    return p

                def thrust_constraint(x_norm):
                    """
                    Logic Rule: produced thrust must meet or exceed
                    the target thrust including rotor self-weight.
                    """
                    self.diameter = x_norm[0] / 10.0
                    self.rpm = x_norm[1] * 1000.0
                    total_req = self.propeller.target_thrust
                    produced = self.propeller.performance["thrust"]
                    return produced - total_req

                constraints = [{'type': 'ineq', 'fun': thrust_constraint}]
                x0_norm = [0.3 * 10.0, 5000 / 1000.0]
                d_max_norm = self.specs['max_diameter'] * 10.0
                bounds_norm = [(0.5, d_max_norm), (0.50, 12.0)]

                res = minimize(
                    obj, x0_norm,
                    method='SLSQP',
                    bounds=bounds_norm,
                    constraints=constraints,
                    options={'ftol': 1e-3}
                )

                if res.success:
                    final_d = res.x[0] / 10.0
                    final_rpm = res.x[1] * 1000.0

                    self.diameter, self.rpm = final_d, final_rpm
                    actual_t = self.propeller.performance["thrust"]

                    if res.fun < best_res["power"]:
                        best_res = {
                            "power": res.fun,
                            "D": final_d,
                            "RPM": final_rpm,
                            "AF": af,
                            "NB": nb,
                            "thrust": actual_t
                        }
                        print(f"*** NEW GLOBAL BEST ***")

        # Domain Protection: raise if no feasible design was found
        if best_res["power"] == float('inf'):
            raise RuntimeError(
                "Optimization failed — no feasible design found. "
                "Consider relaxing constraints or expanding the "
                "airfoil and blade candidate search space."
            )

        # Apply best values back to model BEFORE returning so the
        # ParaPy geometry reflects the optimal design, not the last
        # optimizer iteration
        self.diameter = best_res["D"]
        self.rpm = best_res["RPM"]
        self.propeller.airfoil_type = best_res["AF"]
        self.propeller.n_blades = best_res["NB"]

        print(
            f"\n{'=' * 60}\n"
            f"OPTIMIZATION COMPLETE\n"
            f"Best: NACA {best_res['AF']} | {best_res['NB']} blades | "
            f"D={best_res['D']:.3f}m | RPM={best_res['RPM']:.0f} | "
            f"Power={best_res['power']:.2f}W | "
            f"Thrust={best_res['thrust']:.2f}N\n"
            f"{'=' * 60}"
        )

        return best_res

    @Attribute
    def motor_database(self):
        """
        Integration Rule: reads motors.csv and returns a list of
        dicts, one per motor. Validates file existence before reading.
        """
        if not os.path.exists(self.motor_db_path):
            raise FileNotFoundError(
                f"Motor database not found at '{self.motor_db_path}'. "
                f"Expected location relative to project root. "
                f"Check that motors.csv exists in data/input/."
            )
        motors = []
        with open(self.motor_db_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                motors.append({
                    "name"       : row["name"],
                    "kv"         : float(row["kv"]),
                    "max_power"  : float(row["max_power_w"]),
                    "max_current": float(row["max_current_a"]),
                    "resistance" : float(row["resistance_mohm"]),
                    "mass"       : float(row["mass_g"]),
                })
        return motors

    @Attribute
    def candidate_motors(self):
        """
        Integration Rule: creates an ElectricMotor object for every
        motor in the database and evaluates feasibility against the
        current propeller operating point.
        """
        candidates = []
        for m in self.motor_database:
            motor = ElectricMotor(
                kv          = m["kv"],
                max_power   = m["max_power"],
                max_current = m["max_current"],
                resistance  = m["resistance"],
                mass        = m["mass"],
                rpm_req     = self.propeller.rpm,
                torque_req  = self.propeller.total_torque
            )
            candidates.append((m["name"], motor))
        return candidates

    @Attribute
    def feasible_motors(self):
        """
        Logic Rule: filters candidate motors to only those that
        meet all feasibility constraints.
        """
        return [
            (name, motor)
            for name, motor in self.candidate_motors
            if motor.is_feasible
        ]

    @Attribute
    def best_motor(self):
        """
        Logic Rule: ranks feasible motors by efficiency and returns
        the best one. Raises RuntimeError if no feasible motor found
        with actionable suggestions for the user.
        """
        if not self.feasible_motors:
            raise RuntimeError(
                f"No feasible motor found from "
                f"{len(self.candidate_motors)} candidates. "
                f"Required RPM: {self.propeller.rpm:.0f}, "
                f"Required torque: {self.propeller.total_torque:.3f} Nm. "
                f"Consider: reducing RPM, increasing propeller diameter, "
                f"or adding higher-rated motors to the database."
            )
        return max(self.feasible_motors, key=lambda x: x[1].efficiency)

    @Part(parse=False)
    def motor(self):
        """
        Configuration Rule: instantiates the best feasible motor as
        a proper ParaPy Part so it appears in the model tree.
        parse=False prevents ParaPy from evaluating this Part during
        model tree construction before best_motor is resolved.
        """
        best_name, best = self.best_motor
        return ElectricMotor(
            kv          = best.kv,
            max_power   = best.max_power,
            max_current = best.max_current,
            resistance  = best.resistance,
            mass        = best.mass,
            rpm_req     = self.propeller.rpm,
            torque_req  = self.propeller.total_torque,
            label       = best_name,
        )

    def generate_report(self):
        """
        Prints a summary of the optimal propulsion design.
        Called explicitly from main.py after optimization completes.
        Implemented as a regular method to avoid @Attribute side effects.
        """
        opt = self.run_optimization()
        print(
            f"--- OPTIMAL UAV PROPULSION DESIGN ---\n"
            f"Airfoil: NACA {opt['AF']} | Blades: {opt['NB']}\n"
            f"Diameter: {opt['D']:.3f} m | RPM: {opt['RPM']:.0f}\n"
            f"Power: {opt['power']:.2f} W | Thrust: {opt['thrust']:.2f} N"
        )
        return opt