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

    #: optional input slot — current airfoil type during optimization
    airfoil_type = Input("0012")

    #: optional input slot — current blade count during optimization
    n_blades = Input(2)

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
            safety_margin = self.specs['safety_margin'],
            airfoil_type  = self.airfoil_type,
            n_blades      = self.n_blades
        )


    def _invalidate_propeller_cache(self):
        """
        Forces ParaPy to recalculate propeller geometry by invalidating
        cached attributes after airfoil_type or n_blades changes.
        This is necessary because changing discrete variables (airfoil, blade count)
        invalidates dependent aerospace calculations (splines, airfoil polars, etc.)
        but ParaPy caches them aggressively.
        """
        try:
            # Completely invalidate the propeller Part to force recreating all children
            self.propeller.invalidate()
        except Exception:
            pass

        try:
            # Also try to delete the propeller Part cache if it exists
            if hasattr(self, '_propeller'):
                del self._propeller
        except Exception:
            pass

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
                self.airfoil_type = af
                self.n_blades = nb

                # Invalidate cache after changing airfoil or blade count
                self._invalidate_propeller_cache()

                print(f"\n{'=' * 60}")
                print(f"SEARCHING: NACA {af} | Blades: {nb} "
                      f"| Target Thrust: {req_t:.2f} N")
                print(f"{'-' * 60}")

                def obj(x_norm):
                    """Objective: minimise shaft power."""
                    self.diameter = x_norm[0] / 10.0
                    self.rpm = x_norm[1] * 1000.0
                    perf = self._compute_lightweight_performance()
                    p = perf["shaft_power"]
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
                    # Compute target thrust without triggering mass calculation
                    # target_thrust = base_thrust + mass * 9.81 * safety_margin
                    # Use estimated mass for lightweight calculation
                    estimated_mass = 0.5  # kg, rough estimate for propeller
                    total_req = (self.propeller.base_thrust +
                                 estimated_mass * 9.81 * self.propeller.safety_margin)
                    perf = self._compute_lightweight_performance()
                    produced = perf["thrust"]
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
                    perf = self._compute_lightweight_performance()
                    actual_t = perf["thrust"]

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
        self.airfoil_type = best_res["AF"]
        self.n_blades = best_res["NB"]

        # Force propeller to recalculate with final values
        self._invalidate_propeller_cache()

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
        Assumes run_optimization() has already been called.
        Implemented as a regular method to avoid @Attribute side effects.
        """
        # Get current optimized values (assumes run_optimization was called first)
        perf = self.propeller.performance
        opt = {
            "AF": self.airfoil_type,
            "NB": self.n_blades,
            "D": self.diameter,
            "RPM": self.rpm,
            "power": perf["shaft_power"],
            "thrust": perf["thrust"]
        }
        print(
            f"--- OPTIMAL UAV PROPULSION DESIGN ---\n"
            f"Airfoil: NACA {opt['AF']} | Blades: {opt['NB']}\n"
            f"Diameter: {opt['D']:.3f} m | RPM: {opt['RPM']:.0f}\n"
            f"Power: {opt['power']:.2f} W | Thrust: {opt['thrust']:.2f} N"
        )
        return opt

    def _compute_lightweight_performance(self):
        """
        Computes rotor thrust and power WITHOUT creating ParaPy blade Parts.
        Uses BladeSection calculations directly to avoid triggering Sequence caching.
        Returns dict with 'thrust', 'torque', 'shaft_power'.
        """
        from .blade_section import BladeSection
        import math
        import numpy as np

        try:
            # Recreate sections on-the-fly without via ParaPy Parts
            prop = self.propeller
            n_seg = prop.n_segments

            total_thrust = 0.0
            total_torque = 0.0

            # Manually compute section aerodynamics
            for i in range(n_seg):
                # Create section just for computation, not as a ParaPy Part
                section_data = {
                    'index': i,
                    'parent': prop,
                    'n_segments': n_seg
                }

                # Approximate radius and dr
                dr_approx = (prop.diameter / 2 - 0.02) / n_seg
                r_approx = 0.02 + (i + 0.5) * dr_approx

                # Approximate chord from splines (without creating a Section Part)
                c_spline, p_spline = prop.splines
                chord_approx = float(c_spline(r_approx))
                pitch_approx = float(p_spline(r_approx))

                # Compute section performance
                omega = prop.rpm * 2 * math.pi / 60
                air_density = 1.225
                v_rot = omega * r_approx
                r_tip = prop.diameter / 2
                # Use design_thrust (estimated mass) to avoid triggering blade Part creation
                v_ax = math.sqrt(
                    prop.design_thrust / (2.0 * air_density * math.pi * r_tip ** 2)
                )

                phi = math.atan2(v_ax, v_rot)
                v_eff = math.sqrt(v_ax ** 2 + v_rot ** 2)

                # Get airfoil Cl/Cd
                cl, cd = prop.airfoil.get_cl_cd(pitch_approx - phi)

                # Compute section loads
                l_prime = 0.5 * air_density * v_eff ** 2 * chord_approx * cl
                d_prime = 0.5 * air_density * v_eff ** 2 * chord_approx * cd

                # Per-section contribution (times n_blades internally)
                dT = (l_prime * math.cos(phi) - d_prime * math.sin(phi)) * prop.n_blades * dr_approx
                dQ = (l_prime * math.sin(phi) + d_prime * math.cos(phi)) * r_approx * prop.n_blades * dr_approx

                total_thrust += dT
                total_torque += dQ

            omega = prop.rpm * 2 * math.pi / 60
            shaft_power = total_torque * omega

            return {
                "thrust": total_thrust,
                "torque": total_torque,
                "shaft_power": shaft_power
            }
        except Exception as e:
            # Fallback to using ParaPy Parts if lightweight calculation fails
            print(f"Lightweight performance calculation failed: {e}")
            return self.propeller.performance


