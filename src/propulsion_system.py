import csv
import math
import os

from parapy.core import Base, Input, Attribute, Part, action
from scipy.optimize import minimize

from .motor import ElectricMotor
from .propeller import Propeller


class PropulsionSystem(Base):
    """
    Root class — couples the propeller design with motor selection.
    Reads mission specifications from CSV input, runs a hybrid
    discrete-continuous optimization to find the optimal propeller
    design, reads the motor database and selects the best feasible
    motor match.
    """

    #: required input slot — maximum take-off weight [kg]
    MTOW = Input(5.0)

    #: required input slot — number of rotors on the vehicle
    n_rotors = Input(4)

    #: required input slot — thrust safety margin multiplier
    safety_margin = Input(1.5)

    #: required input slot — maximum allowable propeller diameter [m]
    max_diameter = Input(0.4)

    #: optional input slot — path to motor database CSV file
    motor_db_path = Input("data/input/motors.csv")

    #: optional input slot — initial/current propeller diameter [m]
    #: updated by optimization to optimal value
    diameter = Input(0.3)

    #: optional input slot — initial/current rotational speed [RPM]
    #: updated by optimization to optimal value
    rpm = Input(5000.0)

    #: optional input slot — active number of blades [-]
    #: the optimizer mutates this (not propeller.n_blades) so the change
    #: flows through the @Part binding below, which ParaPy reliably
    #: tracks. Direct mutation on the child Part does not always
    #: invalidate downstream attribute caches with parse=False Parts.
    n_blades = Input(2)

    #: optional input slot — active NACA airfoil code [-]
    #: same propagation rationale as n_blades.
    airfoil_type = Input("4412")

    #: optional input slot — NACA airfoil candidates to search over
    airfoil_candidates = Input(["4412"])    # reduce no of airfoils for testing Input(["0012", "2412", "4412", "6412","2415", "4415", "23012", "23015"])

    #: optional input slot — blade count candidates to search over
    blade_candidates = Input([2, 3, 4])      # , 5, 6, 7, 8, 9, 10])

    #: optional input slot — objective weight for rotor mass [W/kg]
    mass_weight = Input(40)

    #: optional input slot — max tip speed [m/s]
    tip_speed_max = Input(200.0)

    #: optional input slot — speed of sound [m/s]
    speed_of_sound = Input(343.0)

    @Attribute
    def thrust_required(self):
        """
        Mathematical Rule: required thrust per rotor [N].
        Derived from MTOW and number of rotors.
        """
        return self.MTOW * 9.81 / self.n_rotors

    @Part(parse=False)
    def propeller(self):
        """
        Configuration Rule: instantiates the Propeller object with
        mission-derived thrust requirement and the active diameter,
        RPM, n_blades and airfoil_type. All four flow through the Part
        binding so ParaPy properly invalidates downstream caches when
        the optimizer mutates them.
        """
        return Propeller(
            base_thrust   = self.thrust_required,
            diameter      = self.diameter,
            rpm           = self.rpm,
            n_blades      = self.n_blades,
            airfoil_type  = self.airfoil_type,
            safety_margin = self.safety_margin,
            n_segments    = 50,
        )

    def _set_design_point(self, x_norm):
        """
        Helper: apply (diameter, rpm) from normalised SLSQP coordinates.
        Skips the assignment if the value is already current — avoids
        spurious ParaPy invalidation during SLSQP finite-difference
        gradient probes.
        """
        new_d = x_norm[0] / 10.0
        new_r = x_norm[1] * 1000.0
        if self.diameter != new_d:
            self.diameter = new_d
        if self.rpm != new_r:
            self.rpm = new_r

    def _evaluate(self, x_norm):
        """
        Cached propeller evaluation at the current design point. SLSQP
        calls _obj and _thrust_constraint at the same x; this memo
        collapses the pair to one BEMT pass. Keyed on exact float
        (D, RPM) plus n_blades and airfoil — exact equality is essential
        because SLSQP's finite-difference gradient probes use a step of
        ~1.5e-8, so rounding the key would make neighboring probes hit
        the same cache entry and the optimizer would see a zero
        gradient and quit at the initial point.
        """
        self._set_design_point(x_norm)
        key = (self.diameter, self.rpm,
               int(self.n_blades), str(self.airfoil_type))
        cache = getattr(self, "_eval_cache", None)
        if cache is None:
            cache = {}
            self._eval_cache = cache
        hit = cache.get(key)
        if hit is not None:
            return hit
        perf = self.propeller.performance
        result = {
            "power" : perf["shaft_power"],
            "thrust": perf["thrust"],
            "mass"  : self.propeller.mass,
        }
        cache[key] = result
        return result

    def _obj(self, x_norm):
        """
        Objective function for scipy optimizer.
        Minimises shaft power plus a mass penalty (mass_weight [W/kg])
        for a given normalised [diameter, RPM].
        """
        ev    = self._evaluate(x_norm)
        power = ev["power"]
        mass  = ev["mass"]
        obj   = power + self.mass_weight * mass
        print(f"   Iter -> D: {self.diameter:.3f}m "
              f"| RPM: {self.rpm:.0f} "
              f"| P: {power:.1f}W "
              f"| m: {mass * 1000:.1f}g")
        return obj

    def _thrust_constraint(self, x_norm):
        """
        Logic Rule: produced thrust must meet or exceed design thrust.
        Constraint is satisfied when return value >= 0.
        """
        ev       = self._evaluate(x_norm)
        produced = ev["thrust"]
        required = ((self.thrust_required + ev["mass"] * 9.81)
                    * self.safety_margin)
        return produced - required

    def _tip_speed_constraint(self, x_norm):
        self._set_design_point(x_norm)
        omega = self.rpm * 2.0 * math.pi / 60.0
        v_tip = omega * (self.diameter / 2.0)
        return self.tip_speed_max - v_tip

    @Attribute
    def design_summary(self):
        """
        Logic Rule: single tree-visible dict summarising the current
        design. Reactive — recomputes whenever inputs or geometry
        change. Use from the GUI as an at-a-glance status view.
        """
        perf  = self.propeller.performance
        check = self.thrust_check
        health = self.propeller.aero_health_summary
        omega = self.rpm * 2.0 * math.pi / 60.0
        mach_tip = (omega * self.diameter / 2.0) / self.speed_of_sound
        healthy_pct = (100.0 * health["healthy_count"]
                       / max(1, health["total_sections"]))
        summary = {
            "airfoil"          : self.airfoil_type,
            "n_blades"         : int(self.n_blades),
            "diameter_m"       : self.diameter,
            "rpm"              : self.rpm,
            "thrust_N"         : perf["thrust"],
            "torque_Nm"        : perf["torque"],
            "power_W"          : perf["shaft_power"],
            "rotor_mass_g"     : self.propeller.mass * 1000.0,
            "tip_mach"         : mach_tip,
            "thrust_status"    : check["status"],
            "bemt_healthy_pct" : healthy_pct,
        }
        if self.feasible_motors:
            name, motor = self.best_motor
            summary["motor"]            = name
            summary["motor_efficiency"] = motor.efficiency
        return summary

    @Attribute
    def thrust_check(self):
        """
        Logic Rule: live thrust-feasibility check against the current
        propeller geometry and RPM. Reactive — recomputes whenever any
        input the propeller depends on changes. Use from the GUI to
        verify a design without re-running the optimizer.

        A small absolute tolerance absorbs floating-point noise where
        the SLSQP optimum sits right on the constraint boundary
        (produced ≈ required to many decimals); without it, margins of
        order 1e-10 trigger a false "INFEASIBLE" banner.
        """
        produced = self.propeller.performance["thrust"]
        required = ((self.thrust_required
                     + self.propeller.mass * 9.81)
                    * self.safety_margin)
        margin = produced - required
        tol = 1e-3  # 1 mN — well below SLSQP ftol on a ~22 N target
        ok = margin >= -tol
        status = (f"OK (+{margin:.3f} N margin)" if ok
                  else f"INFEASIBLE: short by {-margin:.3f} N")
        return {"required": required, "produced": produced,
                "margin": margin, "ok": ok, "status": status}

    def _validate_inputs(self):
        """
        Logic Rule: warns about unphysical or out-of-range inputs.
        Does not raise — the user may be mid-edit in the GUI and a
        warning is more useful than a crash. Called at the top of
        run_optimization. Returns the list of warning strings.
        """
        warnings = []
        if self.MTOW <= 0:
            warnings.append(f"MTOW={self.MTOW} kg must be positive")
        elif self.MTOW > 1000:
            warnings.append(f"MTOW={self.MTOW} kg is very high for a UAV")
        if self.n_rotors < 1:
            warnings.append(f"n_rotors={self.n_rotors} must be >= 1")
        if self.safety_margin < 1.0:
            warnings.append(
                f"safety_margin={self.safety_margin} < 1.0 — rotor will be "
                f"under-designed by definition"
            )
        if self.max_diameter <= 0:
            warnings.append(f"max_diameter={self.max_diameter} m must be positive")
        elif self.max_diameter > 2.0:
            warnings.append(
                f"max_diameter={self.max_diameter} m is very large for a UAV"
            )
        if self.mass_weight < 0:
            warnings.append(
                f"mass_weight={self.mass_weight} W/kg is negative — would "
                f"reward heavy designs"
            )
        if self.tip_speed_max <= 0 or self.tip_speed_max >= self.speed_of_sound:
            warnings.append(
                f"tip_speed_max={self.tip_speed_max} m/s is unphysical "
                f"(must be 0 < tip_speed_max < {self.speed_of_sound} m/s)"
            )
        if not self.airfoil_candidates:
            warnings.append("airfoil_candidates is empty — nothing to search over")
        if (not self.blade_candidates
                or any(int(b) < 1 for b in self.blade_candidates)):
            warnings.append(
                "blade_candidates is empty or contains non-positive entries"
            )

        thrust_per_rotor = self.MTOW * 9.81 / max(1, self.n_rotors)
        if thrust_per_rotor > 500:
            warnings.append(
                f"thrust per rotor ({thrust_per_rotor:.1f} N) is in "
                f"high-disk-loading territory — double-check MTOW/n_rotors"
            )

        if warnings:
            print("\n" + "=" * 60)
            print("INPUT VALIDATION WARNINGS:")
            for w in warnings:
                print(f"  ! {w}")
            print("=" * 60)
        return warnings

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

        Call explicitly from main.py after instantiation.
        """
        self._validate_inputs()

        # Fresh memoization cache for this optimization run. Cleared per
        # (af, nb) candidate below since the geometry effectively changes.
        self._eval_cache = {}

        best_res = {"power": float('inf'), "thrust": 0.0}
        req_t    = self.thrust_required

        for af in self.airfoil_candidates:
            for nb in self.blade_candidates:
                # Mutate the PropulsionSystem-level Inputs so the change
                # flows through the @Part binding into self.propeller.
                self.airfoil_type = af
                self.n_blades     = nb
                self._eval_cache  = {}

                print(f"\n{'=' * 60}")
                print(f"SEARCHING: NACA {af} | Blades: {nb} "
                      f"| MTOW thrust/rotor: {req_t:.2f} N")
                print(f"{'-' * 60}")

                x0_norm     = [0.3 * 10.0, 5000 / 1000.0]
                d_max_norm  = self.max_diameter * 10.0
                bounds_norm = [(0.8, d_max_norm), (1.0, 12.0)]
                constraints = [
                    {'type': 'ineq', 'fun': self._thrust_constraint},
                    {'type': 'ineq', 'fun': self._tip_speed_constraint}
                ]

                res = minimize(
                    self._obj,
                    x0_norm,
                    method      = 'SLSQP',
                    bounds      = bounds_norm,
                    constraints = constraints,
                    options     = {'ftol': 1e-3}
                )

                if res.success:
                    final_d   = res.x[0] / 10.0
                    final_rpm = res.x[1] * 1000.0

                    self.diameter, self.rpm = final_d, final_rpm
                    actual_t = self.propeller.performance["thrust"]

                    if res.fun < best_res["power"]:
                        best_res = {
                            "power" : res.fun,
                            "D"     : final_d,
                            "RPM"   : final_rpm,
                            "AF"    : af,
                            "NB"    : nb,
                            "thrust": actual_t
                        }
                        print(f"*** NEW GLOBAL BEST ***")

        # Domain Protection: raise if no feasible design found
        if best_res["power"] == float('inf'):
            raise RuntimeError(
                "Optimization failed — no feasible design found. "
                "Consider relaxing constraints or expanding the "
                "airfoil and blade candidate search space."
            )

        # Apply best values back to model so ParaPy geometry
        # reflects the optimal design, not the last iteration.
        # All four flow through the @Part binding (see propeller).
        self.diameter     = best_res["D"]
        self.rpm          = best_res["RPM"]
        self.airfoil_type = best_res["AF"]
        self.n_blades     = best_res["NB"]
        self._eval_cache  = {}

        print(
            f"\n{'=' * 60}\n"
            f"OPTIMIZATION COMPLETE\n"
            f"Best: NACA {best_res['AF']} | {best_res['NB']} blades | "
            f"D={best_res['D']:.3f}m | RPM={best_res['RPM']:.0f} | "
            f"Power={best_res['power']:.2f}W | "
            f"Thrust={best_res['thrust']:.2f}N\n"
            f"{'=' * 60}"
        )

        check = self.thrust_check
        print(f"Required thrust: {check['required']:.3f} N")
        print(f"Produced thrust: {check['produced']:.3f} N")

        omega = self.rpm * 2.0 * math.pi / 60.0
        v_tip = omega * (self.diameter / 2.0)
        mach_tip = v_tip / self.speed_of_sound
        mach_flag = "  (>0.85, near transonic)" if mach_tip > 0.85 else ""
        print(f"Tip Mach: {mach_tip:.2f}{mach_flag}")

        if not check["ok"]:
            print(
                f"\n{'!' * 60}\n"
                f"WARNING: optimal design does NOT meet thrust requirement.\n"
                f"  Required: {check['required']:.3f} N\n"
                f"  Produced: {check['produced']:.3f} N\n"
                f"  Deficit : {-check['margin']:.3f} N\n"
                f"Consider: raise max_diameter, raise tip_speed_max,\n"
                f"          expand airfoil_candidates, or relax safety_margin.\n"
                f"{'!' * 60}"
            )

        health = self.propeller.aero_health_summary
        print(
            f"\nBEMT health summary for final design:\n"
            f"  Healthy / converged sections:       "
            f"{health['healthy_count']} / {health['total_sections']}\n"
            f"  Sections stalled (alpha clamped):   "
            f"{len(health['stalled_radii'])}\n"
            f"  Sections diverged:                  "
            f"{len(health['diverged_radii'])}\n"
            f"  Sections non-converged:             "
            f"{len(health['non_converged_radii'])}\n"
            f"  Sections skipped (root/tip cutoff): "
            f"{health['skipped_count']}"
        )

        return best_res

    @action(label="Re-run optimization")
    def reoptimize(self):
        """
        GUI Action: re-runs the discrete-continuous optimization using
        the current Input values, then prints the design report.
        Exposed as a right-click menu item on the PropulsionSystem node
        so the user can edit MTOW / n_rotors / safety_margin /
        max_diameter / airfoil_candidates / blade_candidates in the GUI
        and refresh the optimal design without restarting.
        """
        self.run_optimization()
        return self.generate_report()

    @action(label="Reset to defaults")
    def reset_to_defaults(self):
        """
        GUI Action: restore mission and search Inputs to factory
        defaults. Useful after experimenting in the GUI tree.
        Does not re-run optimization — call Re-run optimization after.
        """
        self.MTOW              = 5.0
        self.n_rotors          = 4
        self.safety_margin     = 1.5
        self.max_diameter      = 0.4
        self.mass_weight       = 40
        self.tip_speed_max     = 200.0
        self.airfoil_candidates = ["4412"]
        self.blade_candidates  = [2, 3, 4]
        print("Inputs restored to defaults. Run Re-run optimization to refresh.")

    @action(label="Export design CSV")
    def export_design_csv(self):
        """
        GUI Action: write the current optimal design (mission, geometry,
        performance, motor) plus the spanwise distribution to
        data/output/design_<timestamp>.csv. One header row per group,
        followed by the array of per-section values.
        """
        from datetime import datetime
        out_dir = os.path.join("data", "output")
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"design_{stamp}.csv")

        summary = self.design_summary
        spanwise = self.propeller.spanwise_distribution
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["# Design summary"])
            for k, v in summary.items():
                w.writerow([k, v])
            w.writerow([])
            w.writerow(["# Spanwise distribution"])
            w.writerow(["radius_m", "chord_m", "pitch_deg", "dT_N", "dQ_Nm"])
            for row in spanwise:
                w.writerow([f"{v:.6f}" for v in row])
        print(f"Design exported to {path}")
        return path

    @action(label="Plot spanwise distribution")
    def plot_spanwise(self):
        """
        GUI Action: pop a matplotlib window with chord, pitch and
        sectional thrust/torque along the span. Reads
        propeller.spanwise_distribution.
        """
        import matplotlib.pyplot as plt
        rows = self.propeller.spanwise_distribution
        if not rows:
            print("No spanwise distribution available.")
            return
        r       = [row[0] for row in rows]
        chord   = [row[1] * 1000 for row in rows]      # mm
        pitch   = [row[2] for row in rows]              # deg
        dT      = [row[3] for row in rows]              # N per section
        dQ      = [row[4] * 1000 for row in rows]      # mNm per section

        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        axes[0].plot(r, chord, marker=".")
        axes[0].set_ylabel("chord [mm]")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(r, pitch, marker=".", color="tab:orange")
        axes[1].set_ylabel("pitch [deg]")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(r, dT, marker=".", label="dT [N]")
        axes[2].plot(r, dQ, marker=".", label="dQ [mNm]", color="tab:red")
        axes[2].set_ylabel("section load")
        axes[2].set_xlabel("radius [m]")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        fig.suptitle(
            f"NACA {self.airfoil_type} | {int(self.n_blades)} blades | "
            f"D={self.diameter:.3f} m | RPM={self.rpm:.0f}"
        )
        fig.tight_layout()
        plt.show()

    @action(label="Export STEP files")
    def export_step(self):
        """
        GUI Action: write the propeller assembly (hub, flange, all
        rotated blade surfaces) to a single STEP file in data/output/.
        Open the file in any CAD package supporting AP203/AP214 — the
        hub and flange come through as solids, the blades as surfaces.
        """
        from datetime import datetime
        from parapy.exchange import STEPWriter

        out_dir = os.path.join("data", "output")
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"propeller_{stamp}.step")

        prop = self.propeller
        nodes = [prop.hub, prop.hub_flange]
        for blade in prop.blades:
            nodes.append(blade.rotated_surface)

        STEPWriter(nodes=nodes, filename=path).write()
        print(f"STEP geometry exported to {path}")
        return path

    @action(label="Plot motor curve")
    def plot_motor_curve(self):
        """
        GUI Action: pop a matplotlib window with the selected motor's
        torque-vs-speed curve at the operating voltage, with the
        operating point and 80% current/power envelope marked.
        Approximates the motor as a linear DC model:
            Q(omega) = kt * (V/Kv - Q/kt * R / Kv^2)
        rearranged to Q(omega) = (V - omega/Kv_rad) * kt / R_eff
        where omega = RPM * 2π/60.
        """
        import matplotlib.pyplot as plt
        if not self.feasible_motors:
            print("No feasible motor — nothing to plot.")
            return
        name, motor = self.best_motor
        kv          = motor.kv                 # RPM/V
        R           = motor.resistance / 1000  # Ohms (mOhm -> Ohm)
        kt          = motor.kt                  # Nm/A
        V           = motor.voltage_required    # V at operating point

        rpm_arr   = [self.rpm * f for f in [i * 0.05 for i in range(1, 41)]]
        omega_arr = [r * 2.0 * math.pi / 60.0 for r in rpm_arr]
        kv_rad    = kv * 2.0 * math.pi / 60.0   # rad/s/V
        torque    = [max(0.0, (V - om / kv_rad) * kt / max(R, 1e-6))
                     for om in omega_arr]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(rpm_arr, torque, label=f"{name} @ {V:.1f} V")
        ax.axhline(0.8 * motor.max_current * kt, color="red",
                   linestyle="--", label="80% current limit")
        ax.scatter([self.rpm], [motor.torque_req], color="black",
                   zorder=5, label="operating point")
        ax.set_xlabel("RPM")
        ax.set_ylabel("Torque [Nm]")
        ax.set_title(
            f"Motor: {name}  |  Efficiency: {motor.efficiency:.1%}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plt.show()

    @Attribute
    def motor_database(self):
        """
        Integration Rule: reads motors.csv and returns a list of
        dicts, one per motor. Handles missing values gracefully.
        """
        if not os.path.exists(self.motor_db_path):
            raise FileNotFoundError(
                f"Motor database not found at '{self.motor_db_path}'. "
                f"Expected location relative to project root. "
                f"Check that motors.csv exists in data/input/."
            )

        def parse_float(val, default=0.0):
            """Strip ~ and whitespace, return float or default."""
            if not val: return default
            val = val.strip().replace("~", "").split()[0]
            try:    return float(val)
            except ValueError: return default

        motors = []
        with open(self.motor_db_path, newline="",
                  encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")
            print(f"DEBUG headers: {reader.fieldnames}")  # ← add this
            for row in reader:
                if not row.get("name", "").strip():
                    continue
                motors.append({
                    "name"       : row["name"].strip(),
                    "kv"         : parse_float(row["kv"]),
                    "max_power"  : parse_float(row["max_power_w"]),
                    "max_voltage_lipo": parse_float(row["max_voltage_lipo"]),
                    "max_current": parse_float(row["max_current_a"]),
                    "resistance" : parse_float(row["resistance_mohm"]),
                    "mass"       : parse_float(row["mass_g"]),
                    "diameter"   : parse_float(row["diameter_mm"]) / 1000,
                    "height"     : parse_float(row["height_mm"]) / 1000,
                })

        print(f"DEBUG: Reading from {self.motor_db_path}")
        print(f"DEBUG: Found {len(motors)} motors")
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
                max_voltage_lipo = m["max_voltage_lipo"],
                max_current = m["max_current"],
                resistance  = m["resistance"],
                mass        = m["mass"],
                rpm_req     = self.propeller.rpm,
                torque_req  = self.propeller.total_torque,
                motor_D     = m["diameter"],
                motor_h     = m["height"],
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
        the best one. Raises RuntimeError if no feasible motor found.
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
            max_voltage_lipo = best.max_voltage_lipo,
            max_current = best.max_current,
            resistance  = best.resistance,
            mass        = best.mass,
            rpm_req     = self.propeller.rpm,
            torque_req  = self.propeller.total_torque,
            label       = best_name,
            motor_D     = best.motor_D,
            motor_h      = best.motor_h,
            motor_z_offset = -(self.propeller.hub_height / 2 + best.motor_h / 2),
        )

    def generate_report(self):
        """
        Prints a summary of the optimal propulsion design including
        propeller geometry, motor selection and mass breakdown.
        Called explicitly from main.py after optimization completes.
        Implemented as a regular method to avoid @Attribute side effects.
        """
        perf = self.propeller.performance

        print(
            f"\n--- OPTIMAL UAV PROPULSION DESIGN ---\n"
            f"Airfoil:   NACA {self.propeller.airfoil_type}\n"
            f"Blades:    {self.propeller.n_blades}\n"
            f"Diameter:  {self.diameter:.3f} m\n"
            f"RPM:       {self.rpm:.0f}\n"
            f"Power:     {perf['shaft_power']:.2f} W\n"
            f"Thrust:    {perf['thrust']:.2f} N\n"
            f"Rotor mass:{self.propeller.mass * 1000:.1f} g\n"
        )
        summary = {
            "airfoil": self.propeller.airfoil_type,
            "n_blades": self.propeller.n_blades,
            "diameter": self.diameter,
            "rpm": self.rpm,
            "shaft_power": perf["shaft_power"],
            "thrust": perf["thrust"],
            "rotor_mass": self.propeller.mass,
        }

        if self.feasible_motors:
            name, motor = self.best_motor
            print(
                f"Motor:     {name}\n"
                f"KV:        {motor.kv} RPM/V\n"
                f"Efficiency:{motor.efficiency:.1%}\n"
            )
            print(f"DEBUG motor_D: {motor.motor_D * 1000:.1f}mm")
            print(f"DEBUG motor_h: {motor.motor_h * 1000:.1f}mm")
            summary["motor_name"] = name
            summary["motor_kv"] = motor.kv
            summary["motor_efficiency"] = motor.efficiency

        return summary