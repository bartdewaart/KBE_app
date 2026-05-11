import csv
import math
from parapy.core import Base, Input, Attribute, Part
from scipy.optimize import minimize

from .propeller import Propeller
from .motor import ElectricMotor
from .battery import Battery
from .config import get_value


class PropulsionSystem(Base):
    """
    Root class that couples propeller design with motor selection.
    Coordinates the global optimization loop to find the best UAV propulsion setup.
    """

    # Mission inputs provided from main.py (Excel data)
    specs = Input()
    motor_db_path = Input(get_value("propulsion_system", "motor_db_path", default="data/input/motors.csv"))

    # These inputs are updated by the optimizer to drive the geometry
    diameter = Input(get_value("propulsion_system", "diameter", default=0.3))
    rpm = Input(get_value("propulsion_system", "rpm", default=5000))
    airfoil_type = Input(get_value("propulsion_system", "airfoil_type", default="4412"))
    n_blades = Input(get_value("propulsion_system", "n_blades", default=2))

    # Design space candidates
    airfoil_candidates = get_value(
        "propulsion_system",
        "airfoil_candidates",
        default=["0012", "2412", "4412", "6412", "2415", "4415", "23012", "23015"]
    )
    blade_candidates = get_value("propulsion_system", "blade_candidates", default=[2, 3, 4])

    battery = Input(Battery())

    @Attribute
    def thrust_required(self):
        """Calculates the static thrust needed per rotor including safety margin."""
        return (self.specs['MTOW'] * 9.81 / self.specs['n_rotors']) * self.specs['safety_margin']

    @Part
    def propeller(self):
        """The geometric and aerodynamic propeller instance."""
        return Propeller(
            base_thrust=self.thrust_required,
            diameter=self.diameter,
            rpm=self.rpm,
            airfoil_type=self.airfoil_type,
            n_blades=self.n_blades,
            thrust_to_weight=self.specs['safety_margin']
        )

    @Attribute
    def total_weight(self):
        """
        Calculates the total weight including battery weight.
        """
        return self.mtow + self.battery.weight

    @Attribute
    def endurance(self):
        """
        Calculates the endurance using the battery.
        """
        return self.battery.endurance

    @Attribute
    def global_optimization(self):
        """
        Hybrid optimization loop. 
        Iterates through airfoils/blades and optimizes D/RPM for minimum power.
        """
        best_overall = {"power": float('inf')}
        target_t = self.thrust_required

        for af in self.airfoil_candidates:
            for nb in self.blade_candidates:
                # Set temporary values for this iteration
                self.airfoil_type = af
                self.n_blades = nb

                print(f"Testing: NACA {af} | Blades: {nb}")

                def objective(x_norm):
                    """Minimize shaft power: P = Q * omega."""
                    self.diameter = x_norm[0] / 10.0
                    self.rpm = x_norm[1] * 1000.0
                    perf = self.propeller.performance
                    return perf["torque"] * (self.rpm * math.pi / 30)

                def constraint(x_norm):
                    """Constraint: Produced Thrust >= Target (includes blade mass)."""
                    self.diameter = x_norm[0] / 10.0
                    self.rpm = x_norm[1] * 1000.0
                    # target_thrust includes real-time mass of the sections
                    return self.propeller.performance["thrust"] - self.propeller.target_thrust

                # Normalized starting point and bounds for solver stability
                x0_d = get_value("propulsion_system", "optimizer", "x0_diameter", default=0.3)
                x0_r = get_value("propulsion_system", "optimizer", "x0_rpm", default=5000)
                d_min = get_value("propulsion_system", "optimizer", "diameter_min", default=0.1)
                rpm_min = get_value("propulsion_system", "optimizer", "rpm_min", default=1000)
                rpm_max = get_value("propulsion_system", "optimizer", "rpm_max", default=15000)
                x0 = [x0_d * 10.0, x0_r / 1000.0]
                bounds = [(d_min * 10.0, self.specs['max_diameter'] * 10.0), (rpm_min / 1000.0, rpm_max / 1000.0)]
                cons = {'type': 'ineq', 'fun': constraint}

                ftol = get_value("propulsion_system", "optimizer", "ftol", default=1e-3)
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': ftol})

                if res.success and res.fun < best_overall["power"]:
                    best_overall = {
                        "power": res.fun,
                        "D": res.x[0] / 10.0,
                        "RPM": res.x[1] * 1000.0,
                        "AF": af,
                        "NB": nb,
                        "thrust": self.propeller.performance["thrust"]
                    }
                    print(f"  --> New Best Power: {res.fun:.2f} W | Thrust req.: {self.propeller.target_thrust}, Thrust: {self.propeller.performance["thrust"]:.2f}")

        # Normalize metrics and apply weights
        # weights = self.specs.get('weights', {'endurance': 1, 'efficiency': 1, 'weight': 1})
        # max_endurance = max(self.endurance)
        # max_efficiency = max([m["max_power"] / m["mass"] for m in self.motor_database])
        # min_weight = min([m["mass"] for m in self.motor_database])
        #
        # normalized_metrics = {
        #     'endurance': self.endurance / max_endurance,
        #     'efficiency': self.efficiency / max_efficiency,
        #     'weight': min_weight / self.total_weight
        # }
        # objective = sum(weights[metric] * normalized_metrics[metric] for metric in weights)

        if best_overall.get("D") is not None:
            # Ensure the GUI reflects the best solution, not the last iteration.
            self.diameter = best_overall["D"]
            self.rpm = best_overall["RPM"]
            self.airfoil_type = best_overall["AF"]
            self.n_blades = best_overall["NB"]

        return best_overall

    @Attribute
    def motor_database(self):
        """Parses the CSV motor library."""
        motors = []
        with open(self.motor_db_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized_row = {
                    (key or "").strip().lower(): (value.strip() if isinstance(value, str) else value)
                    for key, value in row.items()
                    if key is not None
                }

                def get_value(*keys):
                    for key in keys:
                        value = normalized_row.get(key)
                        if value not in (None, ""):
                            return value
                    raise KeyError(f"Missing required motor column(s): {', '.join(keys)}")

                motors.append({
                    "name": get_value("name", "motor_name", "model"),
                    "kv": float(get_value("kv", "kv_rpm_per_v", "kv_rating")),
                    "max_power": float(get_value("max_power_w", "max_power")),
                    "max_current": float(get_value("max_current_a", "max_current")),
                    "resistance": float(get_value("resistance_mohm", "resistance")),
                    "mass": float(get_value("mass_g", "mass"))
                })
        return motors

    @Attribute
    def best_motor(self):
        """Selects the most efficient feasible motor for the optimal propeller point."""
        opt = self.global_optimization
        feasible = []
        for m in self.motor_database:
            motor_obj = ElectricMotor(
                kv=m["kv"], max_power=m["max_power"], max_current=m["max_current"],
                resistance=m["resistance"], mass=m["mass"],
                rpm_req=opt['RPM'], torque_req=self.propeller.performance["torque"]
            )
            if motor_obj.is_feasible:
                feasible.append((m["name"], motor_obj))

        return max(feasible, key=lambda x: x[1].efficiency) if feasible else (None, None)

    @Part(parse=False)
    def selected_motor(self):
        """Visual representation of the chosen motor in the tree."""
        name, obj = self.best_motor
        if name:
            return ElectricMotor(kv=obj.kv, max_power=obj.max_power, label=name,
                                 rpm_req=obj.rpm_req, torque_req=obj.torque_req)
        return None

    @Attribute
    def generate_report(self):
        """Runs the optimization and 'pushes' results to inputs to update the GUI."""
        opt = self.global_optimization

        # This update triggers ParaPy's dependency graph to redraw the geometry
        self.diameter = opt['D']
        self.rpm = opt['RPM']
        self.airfoil_type = opt['AF']
        self.n_blades = opt['NB']

        print(f"\n{'=' * 40}\nOPTIMAL DESIGN FOUND\n{'=' * 40}")
        print(f"Airfoil: NACA {opt['AF']} | Blades: {opt['NB']}")
        print(f"Diameter: {opt['D']:.3f} m | RPM: {opt['RPM']:.0f}")
        print(f"Power Required: {opt['power']:.2f} W")
        #if self.best_motor[0]:
            #print(f"Selected Motor: {self.best_motor[0]}")

        return opt
