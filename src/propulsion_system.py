import csv
import math
import os

from parapy.core import Base, Input, Attribute, Part, action
from scipy.optimize import minimize

from .battery import Battery
from .motor import ElectricMotor
from .propeller import Propeller


class PropulsionSystem(Base):
    """
    Root class — couples the propeller design with motor selection.

    Reads mission specifications from CSV input, runs a hybrid
    discrete-continuous optimisation to find the optimal propeller
    design, reads the motor database and selects the best feasible
    motor match.

    Optimisation flow:
        For each (airfoil, blade-count) candidate pair (discrete):
            → SLSQP on (diameter, RPM) to minimise shaft power + mass
              penalty subject to thrust and tip-speed constraints.
        Best (lowest objective) feasible design is applied to the model.
    """

    # ─── Mission inputs ───────────────────────────────────────────────────────

    #: required input slot — payload mass (cargo + airframe, excl. props/motors/battery) [kg]
    payload_mass = Input(2.0)

    #: required input slot — number of rotors on the vehicle
    n_rotors = Input(4)

    #: required input slot — thrust safety margin multiplier
    safety_margin = Input(1.5)

    #: required input slot — maximum allowable propeller diameter [m]
    max_diameter = Input(0.4)

    #: optional input slot — propeller blade material
    #: one of "Carbon Fibre", "Fibreglass", "Aluminium", "PLA"
    propeller_material = Input("Carbon Fibre")

    #: optional input slot — pack-level gravimetric energy density [Wh/kg]
    battery_energy_density = Input(250.0)

    #: optional input slot — pack-to-shaft efficiency [-]
    battery_efficiency = Input(0.85)

    #: optional input slot — hover endurance target [min]
    target_endurance_min = Input(10.0)

    #: optional input slot — path to motor database CSV file
    motor_db_path = Input("data/input/motors.csv")

    # ─── Design-point inputs (mutated by the optimiser) ───────────────────────

    #: optional input slot — current propeller diameter [m]; set by optimiser
    diameter = Input(0.3)

    #: optional input slot — current rotational speed [RPM]; set by optimiser
    rpm = Input(5000.0)

    #: optional input slot — active number of blades [-]
    #: the optimiser mutates this Input (not propeller.n_blades) so the change
    #: flows through the @Part binding below, which ParaPy reliably tracks.
    n_blades = Input(2)

    #: optional input slot — active NACA airfoil code [-]
    #: same propagation rationale as n_blades above.
    airfoil_type = Input("4412")

    # ─── Search-space inputs ──────────────────────────────────────────────────

    #: optional input slot — NACA airfoil candidates to search over
    airfoil_candidates = Input(["4412"])

    #: optional input slot — blade count candidates to search over
    blade_candidates = Input([2, 3, 4])

    # ─── Objective weights (must sum to 1.0) ─────────────────────────────────

    #: optional input slot — weight for shaft power (energy efficiency) [-]
    w_power     = Input(0.4)

    #: optional input slot — weight for rotor blade mass (lightweight) [-]
    w_mass      = Input(0.3)

    #: optional input slot — weight for hover endurance (flight time) [-]
    w_endurance = Input(0.3)

    #: optional input slot — maximum tip speed [m/s]
    tip_speed_max = Input(200.0)

    #: optional input slot — speed of sound [m/s]
    speed_of_sound = Input(343.0)

    # ─── Derived mission quantities ───────────────────────────────────────────

    _MATERIAL_DENSITIES = {   # kg/m³
        "Carbon Fibre": 1600,
        "Fibreglass"  : 1900,
        "Aluminium"   : 2700,
        "PLA"         : 1240,
    }

    # Normalization denominators for the three-term objective.
    # A design exactly at these values contributes 1.0 per unit weight.
    # Values represent order-of-magnitude upper bounds for a small UAV.
    _P_REF = 500.0   # W   — reference shaft power
    _M_REF = 0.100   # kg  — reference single-rotor blade mass

    @Attribute
    def objective_weight_check(self):
        """Logic Rule: validates that w_power + w_mass + w_endurance == 1."""
        total = self.w_power + self.w_mass + self.w_endurance
        if abs(total - 1.0) > 0.01:
            return (f"WARNING: weights sum to {total:.3f} — should be 1.0 "
                    f"(w_power={self.w_power}, w_mass={self.w_mass}, "
                    f"w_endurance={self.w_endurance})")
        return (f"OK  w_power={self.w_power}  w_mass={self.w_mass}  "
                f"w_endurance={self.w_endurance}")

    @Attribute
    def payload_weight(self):
        """Mathematical Rule: payload as a force [N]."""
        return self.payload_mass * 9.81

    @Attribute
    def material_density(self):
        """Logic Rule: kg/m³ for the selected propeller_material."""
        try:
            return self._MATERIAL_DENSITIES[self.propeller_material]
        except KeyError:
            valid = ", ".join(self._MATERIAL_DENSITIES)
            raise ValueError(
                f"Unknown propeller_material '{self.propeller_material}'. "
                f"Valid: {valid}"
            )

    @Attribute
    def payload_weight_per_rotor(self):
        """
        Mathematical Rule: payload weight carried by each rotor [N].

        Used as the propeller's base thrust design input.  The propeller adds
        its own estimated blade mass on top (via Propeller.design_thrust), so
        rotor self-weight does not need to appear here.  Battery and motor
        masses are excluded to avoid a circular dependency with battery.mass
        (which depends on propeller.performance which depends on this value);
        their contribution is captured in total_mass and thrust_check instead.
        """
        return self.payload_mass * 9.81 / self.n_rotors

    @Attribute
    def total_mass(self):
        """
        Mathematical Rule: full UAV take-off mass [kg].
        Sum of payload, all rotors, all motors, and the battery.
        Evaluated post-optimisation for reporting; not used as the propeller
        design input (which would create a circular dependency).
        """
        rotor_mass    = self.n_rotors * self.propeller.mass
        motor_mass_kg = (self.n_rotors * self.best_motor[1].mass / 1000.0
                         if self.feasible_motors else 0.0)
        return self.payload_mass + rotor_mass + motor_mass_kg + self.battery.mass

    @Attribute
    def total_electrical_power(self):
        """Mathematical Rule: total electrical power draw across all rotors [W]."""
        shaft_total = self.propeller.performance["shaft_power"] * self.n_rotors
        eta_motor   = self.best_motor[1].efficiency if self.feasible_motors else 0.85
        return shaft_total / max(eta_motor, 1e-3)

    # ─── Propeller part ───────────────────────────────────────────────────────

    @Part(parse=False)
    def propeller(self):
        """
        Configuration Rule: Propeller object wired to the active design
        point.  All four mutable design variables (diameter, rpm, n_blades,
        airfoil_type) flow through this Part binding so ParaPy correctly
        invalidates downstream caches when the optimiser mutates them.
        parse=False defers construction until the optimiser has set the
        design point — avoids evaluating XFOIL with default values at
        instantiation time.
        """
        return Propeller(
            base_thrust      = self.payload_weight_per_rotor,
            diameter         = self.diameter,
            rpm              = self.rpm,
            n_blades         = self.n_blades,
            airfoil_type     = self.airfoil_type,
            safety_margin    = self.safety_margin,
            n_segments       = 50,
            material_density = self.material_density,
        )

    # ─── Optimisation helpers ─────────────────────────────────────────────────

    def _set_design_point(self, x_norm):
        """
        Apply normalised (diameter, RPM) from SLSQP coordinates.

        Normalisation keeps SLSQP gradient steps balanced across the two
        variables, which differ by ~4 orders of magnitude in physical units:
            x[0] = diameter × 10   → typically 0.8–4.0 (order 1)
            x[1] = rpm / 1000      → typically 1.0–12.0 (order 1)

        Skips the assignment when the value is already current — avoids
        spurious ParaPy cache invalidation during SLSQP finite-difference
        gradient probes (step ≈ 1.5e-8).
        """
        new_d = x_norm[0] / 10.0
        new_r = x_norm[1] * 1000.0
        if self.diameter != new_d:
            self.diameter = new_d
        if self.rpm != new_r:
            self.rpm = new_r

    def _evaluate(self, x_norm):
        """
        Cached propeller evaluation at the current design point.

        SLSQP calls _obj and _thrust_constraint at the same x; memoising
        collapses each pair of calls to one BEMT pass.  The cache key uses
        exact floats — rounding would make SLSQP's finite-difference probes
        (step ≈ 1.5e-8) hit the same entry, returning zero gradient and
        causing the optimiser to quit at the starting point.
        """
        self._set_design_point(x_norm)
        key   = (self.diameter, self.rpm,
                 int(self.n_blades), str(self.airfoil_type))
        cache = getattr(self, "_eval_cache", None)
        if cache is None:
            cache = {}
            self._eval_cache = cache
        hit = cache.get(key)
        if hit is not None:
            return hit
        perf   = self.propeller.performance
        result = {
            "power" : perf["shaft_power"],
            "thrust": perf["thrust"],
            "mass"  : self.propeller.mass,
        }
        cache[key] = result
        return result

    def _obj(self, x_norm):
        """
        Objective: normalised weighted sum of power, mass, and endurance.

        All three terms are dimensionless (normalised by _P_REF / _M_REF).
        Endurance proxy = P_REF / P, so lower power → higher endurance term →
        lower objective value (SLSQP minimises, so subtract the endurance term).
        """
        ev   = self._evaluate(x_norm)
        p, m = ev["power"], ev["mass"]

        p_norm = p / self._P_REF
        m_norm = m / self._M_REF
        # endurance: for a fixed battery sized at P_REF × target_endurance_min,
        # flight time ∝ P_REF / P.  Higher value is better → subtract.
        e_norm = self._P_REF / max(p, 1.0)

        obj = (  self.w_power     * p_norm
               + self.w_mass      * m_norm
               - self.w_endurance * e_norm)

        print(f"   Iter -> D: {self.diameter:.3f} m "
              f"| RPM: {self.rpm:.0f} "
              f"| P: {p:.1f} W "
              f"| m: {m * 1000:.1f} g "
              f"| obj: {obj:.4f}")
        return obj

    def _thrust_constraint(self, x_norm):
        """Inequality constraint: produced thrust ≥ design thrust. Returns ≥ 0 when satisfied."""
        ev       = self._evaluate(x_norm)
        produced = ev["thrust"]
        required = self.payload_weight_per_rotor * self.safety_margin
        return produced - required

    def _tip_speed_constraint(self, x_norm):
        """Inequality constraint: tip speed ≤ tip_speed_max. Returns ≥ 0 when satisfied."""
        self._set_design_point(x_norm)
        omega = self.rpm * 2.0 * math.pi / 60.0
        v_tip = omega * (self.diameter / 2.0)
        return self.tip_speed_max - v_tip

    def _search_one_combo(self, af, nb, req_t):
        """
        Run SLSQP for a single (airfoil, blade-count) pair.

        Mutates airfoil_type and n_blades on self so the change flows
        through the propeller @Part binding.  Returns a result dict
        {power, D, RPM, AF, NB, thrust} on success, or None if SLSQP
        did not converge to a feasible solution.
        """
        self.airfoil_type = af
        self.n_blades     = nb
        self._eval_cache  = {}

        print(f"\n{'=' * 60}")
        print(f"SEARCHING: NACA {af} | Blades: {nb} "
              f"| Thrust/rotor: {req_t:.2f} N")
        print(f"{'-' * 60}")

        x0_norm     = [0.3 * 10.0, 5000 / 1000.0]
        d_max_norm  = self.max_diameter * 10.0
        bounds_norm = [(0.8, d_max_norm), (1.0, 12.0)]
        constraints = [
            {"type": "ineq", "fun": self._thrust_constraint},
            {"type": "ineq", "fun": self._tip_speed_constraint},
        ]

        res = minimize(
            self._obj,
            x0_norm,
            method      = "SLSQP",
            bounds      = bounds_norm,
            constraints = constraints,
            options     = {"ftol": 1e-3},
        )

        if not res.success:
            return None

        final_d   = res.x[0] / 10.0
        final_rpm = res.x[1] * 1000.0
        self.diameter, self.rpm = final_d, final_rpm
        actual_t = self.propeller.performance["thrust"]
        return {
            "power"  : res.fun,
            "D"      : final_d,
            "RPM"    : final_rpm,
            "AF"     : af,
            "NB"     : nb,
            "thrust" : actual_t,
        }

    def _apply_best(self, best_res):
        """
        Push the globally-optimal design point back onto the model.

        Sets diameter, rpm, airfoil_type and n_blades so all downstream
        ParaPy Parts and Attributes reflect the optimal design.  Clears
        the eval cache and forces geometry evaluation so the blades, hub
        and flange are fully built before the GUI opens.
        """
        self.diameter     = best_res["D"]
        self.rpm          = best_res["RPM"]
        self.airfoil_type = best_res["AF"]
        self.n_blades     = best_res["NB"]
        self._eval_cache  = {}

        # Force geometry evaluation so hub, flange and blade surfaces are
        # fully in the scene graph when the GUI opens.
        _prop = self.propeller
        _ = _prop.hub
        _ = _prop.hub_flange
        for _blade in _prop.blades:
            _ = _blade.rotated_surface

    def _print_final_report(self, best_res):
        """Print thrust check, tip-Mach number and BEMT health summary."""
        print(
            f"\n{'=' * 60}\n"
            f"OPTIMIZATION COMPLETE\n"
            f"Best: NACA {best_res['AF']} | {best_res['NB']} blades | "
            f"D={best_res['D']:.3f} m | RPM={best_res['RPM']:.0f} | "
            f"Power={best_res['power']:.2f} W | "
            f"Thrust={best_res['thrust']:.2f} N\n"
            f"{'=' * 60}"
        )

        check = self.thrust_check
        print(f"Required thrust: {check['required']:.3f} N")
        print(f"Produced thrust: {check['produced']:.3f} N")

        omega    = self.rpm * 2.0 * math.pi / 60.0
        mach_tip = (omega * self.diameter / 2.0) / self.speed_of_sound
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

    # ─── Main optimisation entry point ────────────────────────────────────────

    def run_optimization(self):
        """
        Integration Rule: Discrete-Continuous Hybrid Optimisation.

        Outer loop: exhaustive search over (airfoil, blade-count) pairs.
        Inner loop: SLSQP on (diameter, RPM) to minimise shaft power +
        mass penalty subject to thrust and tip-speed constraints.

        Implemented as a regular method (not @Attribute) because it
        intentionally mutates Input slots during the search — a necessary
        deviation from functional style when coupling ParaPy models to
        external optimisers.

        Call explicitly from main.py after instantiation; also wired to
        the "Re-run optimization" GUI action.
        """
        self._validate_inputs()
        self._eval_cache = {}

        best_res = {"power": float("inf"), "thrust": 0.0}
        req_t    = self.payload_weight_per_rotor

        for af in self.airfoil_candidates:
            for nb in self.blade_candidates:
                result = self._search_one_combo(af, nb, req_t)
                if result is not None and result["power"] < best_res["power"]:
                    best_res = result
                    print("*** NEW GLOBAL BEST ***")

        if best_res["power"] == float("inf"):
            raise RuntimeError(
                "Optimization failed — no feasible design found. "
                "Consider relaxing constraints or expanding the "
                "airfoil and blade candidate search space."
            )

        self._apply_best(best_res)
        self._print_final_report(best_res)
        return best_res

    # ─── Live design summary ──────────────────────────────────────────────────

    @Attribute
    def design_summary(self):
        """
        Logic Rule: tree-visible dict summarising the current design.
        Reactive — recomputes whenever inputs or geometry change.
        """
        perf   = self.propeller.performance
        check  = self.thrust_check
        health = self.propeller.aero_health_summary
        omega  = self.rpm * 2.0 * math.pi / 60.0
        mach_tip     = (omega * self.diameter / 2.0) / self.speed_of_sound
        healthy_pct  = (100.0 * health["healthy_count"]
                        / max(1, health["total_sections"]))
        summary = {
            "airfoil"              : self.airfoil_type,
            "n_blades"             : int(self.n_blades),
            "diameter_m"           : self.diameter,
            "rpm"                  : self.rpm,
            "thrust_N"             : perf["thrust"],
            "torque_Nm"            : perf["torque"],
            "power_W"              : perf["shaft_power"],
            "rotor_mass_g"         : self.propeller.mass * 1000.0,
            "tip_mach"             : mach_tip,
            "thrust_status"        : check["status"],
            "bemt_healthy_pct"     : healthy_pct,
            "propeller_material"   : self.propeller_material,
            "payload_mass_kg"      : self.payload_mass,
            "total_mass_kg"        : self.total_mass,
            "battery_mass_kg"      : self.battery.mass,
            "battery_capacity_Wh"  : self.battery.capacity_Wh,
            "endurance_min"        : self.battery.endurance_min,
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
        propeller geometry and RPM.

        A 1 mN tolerance absorbs floating-point noise at the SLSQP
        constraint boundary where produced ≈ required to many decimals;
        without it, margins of order 1e-10 trigger a false "INFEASIBLE".
        """
        produced = self.propeller.performance["thrust"]
        required = self.total_mass * 9.81 / self.n_rotors * self.safety_margin
        margin = produced - required
        tol    = 1e-3   # 1 mN — well below SLSQP ftol on a ~22 N target
        ok     = margin >= -tol
        status = (f"OK (+{margin:.3f} N margin)" if ok
                  else f"INFEASIBLE: short by {-margin:.3f} N")
        return {"required": required, "produced": produced,
                "margin": margin, "ok": ok, "status": status}

    # ─── Input validation ─────────────────────────────────────────────────────

    def _validate_inputs(self):
        """
        Logic Rule: warns about unphysical or out-of-range inputs.
        Does not raise — the user may be mid-edit in the GUI and a
        warning is more useful than a crash.  Called at the top of
        run_optimization.  Returns the list of warning strings.
        """
        warnings = []
        if self.payload_mass <= 0:
            warnings.append(f"payload_mass={self.payload_mass} kg must be positive")
        elif self.payload_mass > 1000:
            warnings.append(f"payload_mass={self.payload_mass} kg is very high for a UAV")
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
        total_w = self.w_power + self.w_mass + self.w_endurance
        if abs(total_w - 1.0) > 0.01:
            warnings.append(
                f"Objective weights sum to {total_w:.3f}, not 1.0 "
                f"(w_power={self.w_power}, w_mass={self.w_mass}, "
                f"w_endurance={self.w_endurance})"
            )
        for _name, _val in [("w_power",     self.w_power),
                            ("w_mass",      self.w_mass),
                            ("w_endurance", self.w_endurance)]:
            if _val < 0 or _val > 1:
                warnings.append(f"{_name}={_val} is outside [0, 1]")
        if self.tip_speed_max <= 0 or self.tip_speed_max >= self.speed_of_sound:
            warnings.append(
                f"tip_speed_max={self.tip_speed_max} m/s is unphysical "
                f"(must be 0 < tip_speed_max < {self.speed_of_sound} m/s)"
            )
        if not self.airfoil_candidates:
            warnings.append("airfoil_candidates is empty — nothing to search over")
        if not self.blade_candidates or any(int(b) < 1
                                            for b in self.blade_candidates):
            warnings.append(
                "blade_candidates is empty or contains non-positive entries"
            )
        if self.propeller_material not in self._MATERIAL_DENSITIES:
            valid = ", ".join(self._MATERIAL_DENSITIES)
            warnings.append(
                f"propeller_material='{self.propeller_material}' is not recognised. "
                f"Valid: {valid}"
            )
        if self.battery_energy_density <= 50:
            warnings.append(
                f"battery_energy_density={self.battery_energy_density} Wh/kg seems "
                f"unrealistically low — typical Li-ion is 150–300 Wh/kg"
            )
        if not (0 < self.battery_efficiency <= 1.0):
            warnings.append(
                f"battery_efficiency={self.battery_efficiency} must be in (0, 1]"
            )
        if self.target_endurance_min <= 0:
            warnings.append(
                f"target_endurance_min={self.target_endurance_min} must be positive"
            )
        thrust_per_rotor = self.payload_mass * 9.81 / max(1, self.n_rotors)
        if thrust_per_rotor > 500:
            warnings.append(
                f"payload thrust per rotor ({thrust_per_rotor:.1f} N) is in "
                f"high-disk-loading territory — double-check payload_mass/n_rotors"
            )
        if warnings:
            print("\n" + "=" * 60)
            print("INPUT VALIDATION WARNINGS:")
            for w in warnings:
                print(f"  ! {w}")
            print("=" * 60)
        return warnings

    # ─── GUI actions ──────────────────────────────────────────────────────────

    @action(label="Re-run optimization")
    def reoptimize(self):
        """
        GUI Action: re-runs the optimisation with the current Input values.
        Edit payload_mass / n_rotors / safety_margin / max_diameter /
        airfoil_candidates / blade_candidates in the GUI tree, then click
        this action to refresh the optimal design without restarting.
        """
        self.run_optimization()
        return self.generate_report()

    @action(label="Reset to defaults")
    def reset_to_defaults(self):
        """
        GUI Action: restore mission and search Inputs to factory defaults.
        Does not re-run optimisation — call Re-run optimization after.
        """
        self.payload_mass           = 2.0
        self.n_rotors               = 4
        self.safety_margin          = 1.5
        self.max_diameter           = 0.4
        self.propeller_material     = "Carbon Fibre"
        self.battery_energy_density = 250.0
        self.battery_efficiency     = 0.85
        self.target_endurance_min   = 10.0
        self.w_power                = 0.4
        self.w_mass                 = 0.3
        self.w_endurance            = 0.3
        self.tip_speed_max          = 200.0
        self.airfoil_candidates     = ["4412"]
        self.blade_candidates       = [2, 3, 4]
        print("Inputs restored to defaults. Run Re-run optimization to refresh.")

    @action(label="Export design CSV")
    def export_design_csv(self):
        """
        GUI Action: write the current design summary and spanwise
        distribution to data/output/design_<timestamp>.csv.
        """
        from datetime import datetime
        out_dir = os.path.join("data", "output")
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path  = os.path.join(out_dir, f"design_{stamp}.csv")

        summary  = self.design_summary
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
        GUI Action: matplotlib window showing chord, pitch and sectional
        thrust/torque distributions along the span.
        """
        import matplotlib.pyplot as plt
        rows = self.propeller.spanwise_distribution
        if not rows:
            print("No spanwise distribution available.")
            return
        r     = [row[0] for row in rows]
        chord = [row[1] * 1000 for row in rows]   # mm
        pitch = [row[2] for row in rows]           # deg
        dT    = [row[3] for row in rows]           # N per section
        dQ    = [row[4] * 1000 for row in rows]   # mNm per section

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
        GUI Action: write the propeller assembly (hub, flange and all
        rotated blade surfaces) to a single STEP file in data/output/.
        Hub and flange export as solids; blades export as surfaces.
        Open the file in any CAD package supporting AP203/AP214.
        """
        from datetime import datetime
        from parapy.exchange import STEPWriter

        out_dir = os.path.join("data", "output")
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path  = os.path.join(out_dir, f"propeller_{stamp}.step")

        prop  = self.propeller
        nodes = [prop.hub, prop.hub_flange]
        for blade in prop.blades:
            nodes.append(blade.rotated_surface)

        STEPWriter(nodes=nodes, filename=path).write()
        print(f"STEP geometry exported to {path}")
        return path

    @action(label="Plot motor curve")
    def plot_motor_curve(self):
        """
        GUI Action: matplotlib window showing the selected motor's
        torque-vs-speed curve at the operating voltage, with the
        operating point and 80 % current/power envelope marked.
        """
        import matplotlib.pyplot as plt
        if not self.feasible_motors:
            print("No feasible motor — nothing to plot.")
            return
        name, motor = self.best_motor
        kv       = motor.kv
        R        = motor.resistance / 1000   # mOhm → Ohm
        kt       = motor.kt
        V        = motor.voltage_required

        rpm_arr   = [self.rpm * f for f in [i * 0.05 for i in range(1, 41)]]
        omega_arr = [r * 2.0 * math.pi / 60.0 for r in rpm_arr]
        kv_rad    = kv * 2.0 * math.pi / 60.0
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
        ax.set_title(f"Motor: {name}  |  Efficiency: {motor.efficiency:.1%}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plt.show()

    # ─── Motor selection ──────────────────────────────────────────────────────

    @Attribute
    def motor_database(self):
        """
        Integration Rule: reads motors.csv and returns a list of dicts,
        one per motor.  Handles missing or malformed values gracefully.
        """
        if not os.path.exists(self.motor_db_path):
            raise FileNotFoundError(
                f"Motor database not found at '{self.motor_db_path}'. "
                f"Check that motors.csv exists in data/input/."
            )

        def parse_float(val, default=0.0):
            """Strip ~ and whitespace, return float or default."""
            if not val:
                return default
            val = val.strip().replace("~", "").split()[0]
            try:
                return float(val)
            except ValueError:
                return default

        motors = []
        with open(self.motor_db_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")
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
        return motors

    @Attribute
    def candidate_motors(self):
        """
        Integration Rule: creates an ElectricMotor object for every entry
        in the motor database and evaluates feasibility against the current
        propeller operating point.
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
        """Logic Rule: filters candidate_motors to those that pass both limits."""
        return [
            (name, motor)
            for name, motor in self.candidate_motors
            if motor.is_feasible
        ]

    @Attribute
    def best_motor(self):
        """Logic Rule: returns the feasible motor with the highest efficiency."""
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
        Configuration Rule: the best feasible motor as a ParaPy Part so
        it appears in the model tree.
        parse=False defers construction until best_motor is resolvable
        (after the optimiser has run and a propeller is in the tree).
        """
        best_name, best = self.best_motor
        return ElectricMotor(
            kv             = best.kv,
            max_power      = best.max_power,
            max_current    = best.max_current,
            resistance     = best.resistance,
            mass           = best.mass,
            rpm_req        = self.propeller.rpm,
            torque_req     = self.propeller.total_torque,
            label          = best_name,
            motor_D        = best.motor_D,
            motor_h        = best.motor_h,
            motor_z_offset = -(self.propeller.hub_height / 2 + best.motor_h / 2),
        )

    @Part(parse=False)
    def battery(self):
        """
        Configuration Rule: Battery Part sized for the endurance target.
        parse=False defers construction until total_electrical_power is
        resolvable (requires an optimised propeller and motor in the tree).
        """
        return Battery(
            power_electrical     = self.total_electrical_power,
            target_endurance_min = self.target_endurance_min,
            energy_density       = self.battery_energy_density,
            efficiency           = self.battery_efficiency,
        )

    # ─── Report ───────────────────────────────────────────────────────────────

    def generate_report(self):
        """
        Print a summary of the optimal propulsion design including propeller
        geometry, motor selection and mass breakdown.
        Called explicitly from main.py after optimisation completes.
        Implemented as a regular method (not @Attribute) to avoid the
        side-effect printing running on every tree access.
        """
        perf = self.propeller.performance

        print(
            f"\n--- OPTIMAL UAV PROPULSION DESIGN ---\n"
            f"Airfoil:      NACA {self.propeller.airfoil_type}\n"
            f"Blades:       {self.propeller.n_blades}\n"
            f"Diameter:     {self.diameter:.3f} m\n"
            f"RPM:          {self.rpm:.0f}\n"
            f"Power:        {perf['shaft_power']:.2f} W\n"
            f"Thrust:       {perf['thrust']:.2f} N\n"
            f"Material:     {self.propeller_material}\n"
            f"Rotor mass:   {self.propeller.mass * 1000:.1f} g × {self.n_rotors} rotors\n"
        )
        print(
            f"--- MASS BUDGET ---\n"
            f"Payload:      {self.payload_mass * 1000:.0f} g\n"
            f"Rotors:       {self.n_rotors * self.propeller.mass * 1000:.0f} g\n"
        )

        summary = {
            "airfoil"          : self.propeller.airfoil_type,
            "n_blades"         : self.propeller.n_blades,
            "diameter"         : self.diameter,
            "rpm"              : self.rpm,
            "shaft_power"      : perf["shaft_power"],
            "thrust"           : perf["thrust"],
            "rotor_mass"       : self.propeller.mass,
            "propeller_material": self.propeller_material,
            "payload_mass_kg"  : self.payload_mass,
        }

        if self.feasible_motors:
            name, motor = self.best_motor
            print(
                f"Motor:        {name}\n"
                f"KV:           {motor.kv} RPM/V\n"
                f"Efficiency:   {motor.efficiency:.1%}\n"
                f"Motor mass:   {motor.mass:.0f} g × {self.n_rotors} rotors\n"
            )
            summary["motor_name"]       = name
            summary["motor_kv"]         = motor.kv
            summary["motor_efficiency"] = motor.efficiency

        print(
            f"Battery:      {self.battery.mass * 1000:.0f} g  "
            f"({self.battery.capacity_Wh:.1f} Wh)\n"
            f"Endurance:    {self.battery.endurance_min:.1f} min\n"
            f"Total mass:   {self.total_mass * 1000:.0f} g\n"
        )
        summary["battery_mass_kg"]     = self.battery.mass
        summary["battery_capacity_Wh"] = self.battery.capacity_Wh
        summary["endurance_min"]       = self.battery.endurance_min
        summary["total_mass_kg"]       = self.total_mass

        return summary
