import csv
import math
import os
import re

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

    #: optional input slot — minimum required hover endurance [min]
    #: acts as a hard floor: SLSQP rejects designs that cannot achieve it;
    #: w_endurance then pushes the optimizer above this floor.
    min_endurance_min = Input(10.0)

    #: optional input slot — minimum structural factor of safety [-]
    #: used to flag designs where blade stress exceeds material limits
    structural_fos_min = Input(2.0)

    #: optional input slot — battery energy budget [Wh]; set by optimiser.
    #: Do not edit manually — run Re-run optimization to update.
    battery_capacity_Wh = Input(0.0)

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
    airfoil_candidates = Input(["4412", "2412", "6412", "4409", "4415", "0012", "0009"])

    #: optional input slot — blade count candidates to search over
    blade_candidates = Input([2, 3, 4])

    # ─── Objective weights (must sum to 1.0) ─────────────────────────────────

    #: optional input slot — weight for shaft power (energy efficiency) [-]
    w_power     = Input(0.5)

    #: optional input slot — weight for rotor blade mass [-]
    #: Default 0.0 — blade mass is negligible vs battery+motors and its gradient
    #: pushes the optimizer toward unrealistically thin blades.  The thrust
    #: constraint already penalises heavy blades; set non-zero only to
    #: explicitly minimise rotor weight (e.g. for coaxial / micro UAVs).
    w_mass      = Input(0.0)

    #: optional input slot — weight for hover endurance (flight time) [-]
    w_endurance = Input(0.5)

    #: optional input slot — maximum tip speed [m/s]
    tip_speed_max = Input(200.0)

    #: optional input slot — speed of sound [m/s]
    speed_of_sound = Input(343.0)

    #: optional input slot — air density [kg/m³]
    #: propagated to Propeller and BladeSection; adjust for altitude
    air_density = Input(1.225)

    #: optional input slot — target tip speed for the SLSQP starting point [m/s]
    #: sets the initial RPM guess; 60–80 m/s is the efficient hover range for small rotors
    initial_tip_speed = Input(65.0)

    # ─── Derived mission quantities ───────────────────────────────────────────

    _MATERIAL_DENSITIES = {   # kg/m³
        "Carbon Fibre": 1600,
        "Fibreglass"  : 1900,
        "Aluminium"   : 2700,
        "PLA"         : 1240,
    }

    _MATERIAL_MECHANICS = {   # E [Pa], sigma_ult [Pa]
        "Carbon Fibre": {"E": 60e9,  "sigma_ult": 600e6},
        "Fibreglass"  : {"E": 20e9,  "sigma_ult": 200e6},
        "Aluminium"   : {"E": 70e9,  "sigma_ult": 270e6},
        "PLA"         : {"E":  3.5e9, "sigma_ult":  60e6},
    }

    # Normalization denominators for the three-term objective.
    # A design exactly at these values contributes 1.0 per unit weight.
    # Values represent order-of-magnitude upper bounds for a small UAV.
    _P_REF = 500.0   # W   — reference shaft power
    _M_REF = 0.100   # kg  — reference single-rotor blade mass

    # Fallback motor efficiency when no feasible motor has been selected yet.
    # Used during SLSQP iterations before motor selection is resolved.
    _DEFAULT_MOTOR_ETA = 0.85

    # Reference system efficiency (prop + motor) used to size the battery
    # and estimate shaft power from ideal actuator-disk power.
    _SYSTEM_ETA_REF = 0.65

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
    def motor_mass_estimate_g(self):
        """
        Mathematical Rule: maximum motor mass in the database [g].
        best_motor selects for highest efficiency, and efficient motors tend
        to be the heaviest in the catalogue.  Using the maximum ensures the
        SLSQP thrust constraint is always at least as conservative as the
        post-optimisation thrust_check regardless of which motor is chosen.
        Falls back to 100 g if the database is missing or empty.
        """
        try:
            masses = [m["mass"] for m in self.motor_database if m["mass"] > 0]
            if masses:
                return max(masses)
        except Exception:
            pass
        return 100.0

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
    def material_E(self):
        """Logic Rule: Young's modulus [Pa] for the selected propeller_material."""
        return self._MATERIAL_MECHANICS[self.propeller_material]["E"]

    @Attribute
    def material_sigma_ult(self):
        """Logic Rule: ultimate tensile strength [Pa] for the selected propeller_material."""
        return self._MATERIAL_MECHANICS[self.propeller_material]["sigma_ult"]

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
                         if self.feasible_motors
                         else self.n_rotors * self.motor_mass_estimate_g / 1000.0)
        return self.payload_mass + rotor_mass + motor_mass_kg + self.battery.mass

    @Attribute
    def total_electrical_power(self):
        """Mathematical Rule: total electrical power draw across all rotors [W]."""
        shaft_total = self.propeller.performance["shaft_power"] * self.n_rotors
        eta_motor   = (self.best_motor[1].efficiency
                       if self.feasible_motors else self._DEFAULT_MOTOR_ETA)
        return shaft_total / max(eta_motor, 1e-3)

    # ─── Propeller part ───────────────────────────────────────────────────────

    @Part(parse=False)
    def propeller(self):
        """Configuration Rule: Propeller wired to the active design point.
        parse=False defers XFOIL evaluation until after the first optimisation run."""
        return Propeller(
            base_thrust        = self.payload_weight_per_rotor,
            diameter           = self.diameter,
            rpm                = self.rpm,
            n_blades           = self.n_blades,
            airfoil_type       = self.airfoil_type,
            safety_margin      = self.safety_margin,
            n_segments         = 50,
            material_density   = self.material_density,
            material_E         = self.material_E,
            material_sigma_ult = self.material_sigma_ult,
            air_density        = self.air_density,
        )

    # ─── Optimisation helpers ─────────────────────────────────────────────────

    @staticmethod
    def _show_popup(title, message, kind="warning"):
        """Show a blocking tkinter dialog. kind = 'warning' | 'info' | 'error'."""
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.lift()
            root.focus_force()
            if kind == "info":
                messagebox.showinfo(title=title, message=message, parent=root)
            elif kind == "error":
                messagebox.showerror(title=title, message=message, parent=root)
            else:
                messagebox.showwarning(title=title, message=message, parent=root)
            root.destroy()
        except Exception:
            pass   # headless / tkinter unavailable — console output already printed

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
        """Memoised BEMT evaluation. Cache key uses exact floats — rounding would
        collapse FD probe steps and zero the gradient, stalling the optimiser."""
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

        # Actual endurance from the fixed battery budget: lower P → longer flight.
        # Divide by total electrical power (shaft → electrical via motor eta fallback).
        # Using shaft power directly would overestimate endurance by ~18%.
        # e_norm = 1.0 at the floor; > 1.0 when the design beats the minimum.
        actual_end = 0.0
        bat_cap = getattr(self, "_bat_cap_Wh", 0.0)
        if bat_cap > 0:
            p_electrical = p * max(self.n_rotors, 1) / self._DEFAULT_MOTOR_ETA
            actual_end = 60.0 * bat_cap * self.battery_efficiency / max(p_electrical, 1e-6)
            e_norm = actual_end / max(self.min_endurance_min, 1.0)
        else:
            e_norm = self._P_REF / max(p, 1.0)   # fallback before first sizing

        obj = (  self.w_power     * p_norm
               + self.w_mass      * m_norm
               - self.w_endurance * e_norm)

        end_str = f"{actual_end:.1f} min" if bat_cap > 0 else "—"
        print(f"   Iter -> D: {self.diameter:.3f} m "
              f"| RPM: {self.rpm:.0f} "
              f"| P: {p:.1f} W "
              f"| m: {m * 1000:.1f} g "
              f"| end: {end_str} "
              f"| obj: {obj:.4f}")
        return obj

    def _thrust_constraint(self, x_norm):
        """
        Inequality constraint: produced thrust ≥ full-UAV design thrust.
        Returns ≥ 0 when satisfied.

        Battery mass is derived inline from ev["power"] rather than via
        self.battery.mass to avoid the circular @Attribute chain
        (battery → total_electrical_power → propeller → base_thrust).
        Motor mass is omitted here (not yet known; small vs battery+payload).
        """
        ev         = self._evaluate(x_norm)
        bat_mass   = (getattr(self, "_bat_cap_Wh", 0.0)
                      / max(self.battery_energy_density, 1e-6))
        motor_mass = self.n_rotors * self.motor_mass_estimate_g / 1000.0
        total_m    = (self.payload_mass
                      + self.n_rotors * ev["mass"]
                      + motor_mass
                      + bat_mass)
        required   = total_m * 9.81 / self.n_rotors * self.safety_margin
        return ev["thrust"] - required

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

        # Start diameter at half the maximum; start RPM from a target tip speed
        # of 65 m/s (efficient hover range for small UAV rotors).  omega is
        # computed at the *initial* diameter's radius (max_diameter/4), not the
        # full max_diameter's radius — using max_diameter/2 was a bug that gave
        # half the intended tip speed at x0, starting SLSQP far from feasible.
        _tip_speed_init = self.initial_tip_speed     # m/s — target hover tip speed
        _x0_radius      = self.max_diameter / 4      # radius at x0 diameter (D=max/2)
        _omega_init     = _tip_speed_init / _x0_radius
        _rpm_init       = _omega_init * 60.0 / (2.0 * math.pi)
        _rpm_init       = max(1500.0, min(10000.0, _rpm_init))  # stay within bounds
        x0_norm    = [self.max_diameter / 2 * 10.0, _rpm_init / 1000.0]
        d_max_norm = self.max_diameter * 10.0
        bounds_norm = [(0.8, d_max_norm), (1.0, 12.0)]

        # Battery energy budget — fixed for the whole SLSQP run so that the
        # endurance objective is a genuine function of design power.
        # Sized at ideal actuator-disk power for the payload at max_diameter,
        # divided by a typical total-system efficiency of 0.65 (prop + motor).
        # This is a conservative lower-bound on real shaft power, ensuring the
        # battery can actually achieve min_endurance_min for efficient designs.
        # Endurance is enforced as a soft post-check (warning), NOT as a hard
        # SLSQP constraint, because adding it as a hard constraint creates a
        # circular infeasibility: the battery mass depends on P_ref, which
        # determines required thrust, which in turn constrains P — the system
        # has no guaranteed feasible interior point from an arbitrary x0.
        _T_ref = self.payload_mass * 9.81 / max(self.n_rotors, 1)
        _A_ref = math.pi * (self.max_diameter / 2) ** 2
        _P_disk = _T_ref ** 1.5 / math.sqrt(2 * self.air_density * max(_A_ref, 1e-6))
        p_ref = _P_disk * self.n_rotors / self._SYSTEM_ETA_REF
        self._bat_cap_Wh = (p_ref * self.min_endurance_min / 60.0
                            ) / max(self.battery_efficiency, 1e-6)
        print(f"   Battery reference: {p_ref:.0f} W system x "
              f"{self.min_endurance_min:.1f} min "
              f"-> {self._bat_cap_Wh:.1f} Wh  "
              f"({self._bat_cap_Wh / max(self.battery_energy_density, 1e-6) * 1000:.0f} g)")

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
            options     = {"ftol": 1e-2, "eps": 1e-2, "maxiter": 60},
        )

        if not res.success:
            return None

        # SLSQP can report success=True while leaving constraints slightly
        # violated (numerical tolerances).  Re-evaluate both constraints
        # at the final point and reject if either is significantly violated.
        thrust_margin = self._thrust_constraint(res.x)
        tip_margin    = self._tip_speed_constraint(res.x)
        if thrust_margin < -1.0 or tip_margin < -1.0:
            print(f"   SLSQP success but constraints violated "
                  f"(thrust margin={thrust_margin:.2f} N, "
                  f"tip margin={tip_margin:.2f} m/s) — discarding.")
            return None

        final_d   = res.x[0] / 10.0
        final_rpm = res.x[1] * 1000.0
        self.diameter, self.rpm = final_d, final_rpm

        if abs(final_d - self.max_diameter) < 0.001:
            print(f"   *** Diameter bound active: D={final_d:.3f} m = max_diameter "
                  f"— consider increasing max_diameter for a better solution.")
        if abs(final_rpm - 12000.0) < 10.0:
            print(f"   *** RPM upper bound active: {final_rpm:.0f} RPM "
                  f"— consider increasing tip_speed_max for a better solution.")

        perf     = self.propeller.performance
        actual_t = perf["thrust"]
        return {
            "objective"  : res.fun,                  # dimensionless SLSQP objective
            "power_W"    : perf["shaft_power"],       # actual shaft power per rotor [W]
            "D"          : final_d,
            "RPM"        : final_rpm,
            "AF"         : af,
            "NB"         : nb,
            "thrust"     : actual_t,
            "bat_cap_Wh" : self._bat_cap_Wh,
        }

    def _apply_best(self, best_res):
        """Push the optimal design point onto Input slots and size the battery
        to actual BEMT shaft power (not the ideal-disk reference used in SLSQP)."""
        self.diameter     = best_res["D"]
        self.rpm          = best_res["RPM"]
        self.airfoil_type = best_res["AF"]
        self.n_blades     = best_res["NB"]
        self._eval_cache  = {}

        # Size battery to guarantee min_endurance_min with the ACTUAL shaft
        # power of the optimised propeller.  best_res["bat_cap_Wh"] was only
        # a reference estimate used inside SLSQP; actual BEMT power is higher
        # than the ideal-disk reference, so using it would under-size the battery.
        actual_shaft = self.propeller.performance["shaft_power"]
        eta_motor    = (self.best_motor[1].efficiency
                        if self.feasible_motors else self._DEFAULT_MOTOR_ETA)
        p_elec_actual = actual_shaft * self.n_rotors / max(eta_motor, 1e-3)
        self.battery_capacity_Wh = (
            p_elec_actual * self.min_endurance_min
            / (60.0 * max(self.battery_efficiency, 1e-6))
        )

        # Warn via popup if the full-mass thrust check fails.
        check = self.thrust_check
        if not check["ok"]:
            self._show_popup(
                title="Thrust Requirement Not Met",
                message=(
                    f"The optimised rotor cannot lift the full UAV.\n\n"
                    f"  Produced:  {check['produced']:.2f} N\n"
                    f"  Required:  {check['required']:.2f} N\n"
                    f"  Shortfall: {-check['margin']:.2f} N\n\n"
                    f"Suggestions:\n"
                    f"  • Increase max_diameter\n"
                    f"  • Reduce payload_mass\n"
                    f"  • Reduce min_endurance_min\n"
                    f"  • Reduce safety_margin\n"
                    f"  • Increase tip_speed_max"
                ),
                kind="warning"
            )
        self._check_final_warnings()

    def _check_final_warnings(self):
        """Post-optimization popup for propeller-level anomalies and bound-active conditions."""
        issues = []
        if self.propeller.total_thrust < 0:
            issues.append(
                f"Total thrust is negative ({self.propeller.total_thrust:.2f} N). "
                f"Pitch angles may be inverted or RPM too low."
            )
        if self.propeller.total_torque < 0:
            issues.append(
                f"Total torque is negative ({self.propeller.total_torque:.2f} Nm). "
                f"Check spline pitch distribution."
            )
        if abs(self.diameter - self.max_diameter) < 0.001:
            issues.append(
                f"Diameter ({self.diameter:.3f} m) hit the max_diameter bound — "
                f"a better solution may exist at larger diameter. "
                f"Consider increasing max_diameter."
            )
        if abs(self.rpm - 12000.0) < 10.0:
            issues.append(
                f"RPM ({self.rpm:.0f}) hit the upper bound (12 000 RPM) — "
                f"consider increasing tip_speed_max."
            )
        try:
            sa = self.propeller.structural_analysis
            if sa and sa["min_FoS"] < self.structural_fos_min:
                issues.append(
                    f"Structural FoS = {sa['min_FoS']:.2f} at "
                    f"r = {sa['critical_radius_m'] * 1000:.0f} mm "
                    f"(minimum allowed: {self.structural_fos_min:.1f}). "
                    f"Consider: stiffer material, larger chord, or fewer blades."
                )
        except Exception:
            pass
        if issues:
            self._show_popup(
                title="Design Warnings",
                message="\n\n".join(f"• {w}" for w in issues),
                kind="warning"
            )

    def _print_final_report(self, best_res):
        """Print thrust check, tip-Mach number and BEMT health summary."""
        print(
            f"\n{'=' * 60}\n"
            f"OPTIMIZATION COMPLETE\n"
            f"Best: NACA {best_res['AF']} | {best_res['NB']} blades | "
            f"D={best_res['D']:.3f} m | RPM={best_res['RPM']:.0f} | "
            f"Power={best_res['power_W']:.2f} W/rotor | "
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
        """Integration Rule: outer exhaustive search over (airfoil, blade-count) pairs;
        inner SLSQP on (diameter, RPM). Mutates Input slots — cannot be an @Attribute."""
        val_warnings = self._validate_inputs()
        if val_warnings:
            self._show_popup(
                title="Input Validation Warnings",
                message=(
                    "The following issues were detected before optimization:\n\n"
                    + "\n".join(f"  • {w}" for w in val_warnings)
                    + "\n\nOptimization will still run — check results carefully."
                ),
                kind="warning"
            )

        self._eval_cache  = {}
        self._bat_cap_Wh  = 0.0   # reset; each combo re-sizes at its x0

        best_res = {"objective": float("inf"), "thrust": 0.0}
        req_t    = self.payload_weight_per_rotor

        for af in self.airfoil_candidates:
            for nb in self.blade_candidates:
                result = self._search_one_combo(af, nb, req_t)
                if result is not None and result["objective"] < best_res["objective"]:
                    best_res = result
                    print("*** NEW GLOBAL BEST ***")

        if best_res["objective"] == float("inf"):
            raise RuntimeError(
                "Optimization failed — no (airfoil, blade-count) combination "
                "produced a converged, feasible design.\nSuggestions:\n"
                f"  • Increase max_diameter (currently {self.max_diameter:.2f} m)\n"
                f"  • Reduce safety_margin (currently {self.safety_margin:.2f})\n"
                f"  • Reduce min_endurance_min "
                f"(currently {self.min_endurance_min:.1f} min)\n"
                f"  • Expand airfoil_candidates "
                f"(currently {self.airfoil_candidates})\n"
                f"  • Expand blade_candidates "
                f"(currently {self.blade_candidates})\n"
                f"  • Increase tip_speed_max "
                f"(currently {self.tip_speed_max:.0f} m/s)"
            )

        self._apply_best(best_res)
        self._print_final_report(best_res)

        check   = self.thrust_check
        end_min = self.battery.endurance_min
        end_ok  = end_min >= self.min_endurance_min
        if check["ok"]:
            end_note = (f"  Endurance: {end_min:.1f} min  ✓"
                        if end_ok else
                        f"  Endurance: {end_min:.1f} min  "
                        f"*** below floor {self.min_endurance_min:.1f} min ***")
            self._show_popup(
                title="Optimization Complete",
                message=(
                    f"Optimal design found.\n\n"
                    f"  Airfoil:   NACA {best_res['AF']}\n"
                    f"  Blades:    {best_res['NB']}\n"
                    f"  Diameter:  {best_res['D']:.3f} m\n"
                    f"  RPM:       {best_res['RPM']:.0f}\n"
                    f"  Power:     {best_res['power_W']:.1f} W/rotor\n"
                    f"  Thrust:    {best_res['thrust']:.2f} N\n"
                    f"{end_note}"
                    + ("" if end_ok else
                       "\n\nTo meet the endurance floor:\n"
                       "  • Increase max_diameter\n"
                       "  • Reduce safety_margin\n"
                       "  • Reduce min_endurance_min\n"
                       "  • Increase battery_energy_density")
                ),
                kind="info" if end_ok else "warning"
            )

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
        tol    = 1e-2   # 1 mN — well below SLSQP ftol on a ~22 N target
        ok     = margin >= -tol
        status = (f"OK (+{margin:.3f} N margin)" if ok
                  else f"INFEASIBLE: short by {-margin:.3f} N")
        return {"required": required, "produced": produced,
                "margin": margin, "ok": ok, "status": status}

    @Attribute
    def thrust_warning(self):
        """
        Logic Rule: human-readable thrust feasibility status shown in the
        GUI tree.  Always visible — red-flag text when infeasible so the
        user sees it without opening design_summary.
        """
        check = self.thrust_check
        if check["ok"]:
            return (f"OK — margin {check['margin']:.2f} N  "
                    f"(produced {check['produced']:.1f} N  "
                    f"/ required {check['required']:.1f} N)")
        return (f"*** THRUST INFEASIBLE ***  "
                f"short by {-check['margin']:.2f} N  "
                f"(produced {check['produced']:.1f} N  "
                f"/ required {check['required']:.1f} N)  "
                f"— raise max_diameter or reduce payload / endurance target")

    @Attribute
    def endurance_warning(self):
        """
        Logic Rule: compares actual battery endurance against the minimum
        required floor.  Visible in the GUI tree as a persistent status.
        Endurance is a soft post-check — the optimiser maximises it via
        w_endurance but does not enforce the floor as a hard constraint
        (doing so creates a circular infeasibility; see _search_one_combo).
        """
        actual = self.battery.endurance_min
        floor  = self.min_endurance_min
        if actual >= floor - 0.05:   # 3-second tolerance for floating-point
            return (f"OK — {actual:.1f} min  "
                    f"(floor = {floor:.1f} min, "
                    f"+{actual - floor:.1f} min margin)")
        return (f"*** BELOW FLOOR: {actual:.1f} min < {floor:.1f} min — "
                f"increase max_diameter, reduce safety_margin, or "
                f"reduce min_endurance_min ***")

    @Attribute
    def bemt_health_warning(self):
        """
        Logic Rule: warns when > 20 % of active BEMT sections are unreliable
        (stalled / diverged / non-converged).  Visible in the GUI tree without
        opening design_summary.
        """
        health = self.propeller.aero_health_summary
        total  = health["total_sections"]
        bad    = (len(health["stalled_radii"])
                  + len(health["diverged_radii"])
                  + len(health["non_converged_radii"]))
        if total == 0:
            return "No BEMT sections evaluated."
        bad_pct = bad / total
        if bad_pct > 0.20:
            return (f"*** BEMT UNRELIABLE: {bad}/{total} sections "
                    f"stalled/diverged/non-converged ({bad_pct:.0%}) — "
                    f"aerodynamic results may be inaccurate ***")
        return (f"OK — {health['healthy_count']}/{total} healthy sections "
                f"({100.0 * health['healthy_count'] / total:.0f}%)")

    @Attribute
    def design_bounds_warning(self):
        """
        Logic Rule: warns when the optimised design is pinned against the
        diameter or RPM search bounds, indicating that a better solution may
        exist beyond the current limits.
        """
        parts = []
        if abs(self.diameter - self.max_diameter) < 0.001:
            parts.append(
                f"diameter ({self.diameter:.3f} m) hit max_diameter bound — "
                f"try increasing max_diameter"
            )
        if abs(self.rpm - 12000.0) < 10.0:
            parts.append(
                f"RPM ({self.rpm:.0f}) hit upper bound 12 000 — "
                f"try increasing tip_speed_max"
            )
        if not parts:
            return "OK — design is interior to both search bounds"
        return "*** BOUNDS ACTIVE: " + "; ".join(parts) + " ***"

    @Attribute
    def motor_warning(self):
        """
        Logic Rule: warns about no feasible motor, low motor efficiency, or
        marginal current headroom (< 10 % of 80 % rated limit).
        Safe to read even when no feasible motor exists.
        """
        if not self.feasible_motors:
            n_cands = len(self.candidate_motors)
            return (f"*** NO FEASIBLE MOTOR: 0 / {n_cands} candidates meet "
                    f"RPM={self.rpm:.0f} / torque={self.propeller.total_torque:.3f} Nm. "
                    f"Add higher-rated motors to motors.csv or reduce RPM. ***")
        name, motor = self.best_motor
        parts = []
        if motor.efficiency < 0.5:
            parts.append(
                f"low efficiency ({motor.efficiency:.1%}) for '{name}' — "
                f"check motor database values"
            )
        report = motor.feasibility_report
        if report.get("marginal", False):
            parts.append(
                f"marginal current headroom "
                f"({report['current_margin']:.2f} A, < 10 % of limit)"
            )
        if not parts:
            return f"OK — '{name}'  efficiency {motor.efficiency:.1%}"
        return "WARNING: " + "; ".join(parts)

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
        _hub_r = 0.02   # proxy hub radius [m]
        if (self.max_diameter / 2) <= _hub_r:
            warnings.append(
                f"max_diameter={self.max_diameter} m is too small — blade span "
                f"(R − hub_radius) ≤ 0 (hub proxy = {_hub_r} m)"
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
        for _af in self.airfoil_candidates:
            if not re.match(r'^\d{4,5}$', str(_af)):
                warnings.append(
                    f"airfoil code '{_af}' does not look like a 4- or 5-digit "
                    f"NACA code (expected e.g. '4412' or '23012')"
                )
        if not self.blade_candidates or any(int(b) < 1
                                            for b in self.blade_candidates):
            warnings.append(
                "blade_candidates is empty or contains non-positive entries"
            )
        if self.blade_candidates and any(int(b) > 8 for b in self.blade_candidates):
            _heavy = [b for b in self.blade_candidates if int(b) > 8]
            warnings.append(
                f"blade_candidates contains values > 8 {_heavy} — "
                f"BEMT accuracy degrades significantly above 6–8 blades"
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
        if self.max_diameter > 0 and self.n_rotors >= 1:
            _area    = math.pi * (self.max_diameter / 2) ** 2
            _t_rotor = self.payload_mass * 9.81 / self.n_rotors
            _p_disk  = (_t_rotor ** 1.5) / math.sqrt(2 * self.air_density * max(_area, 1e-6))
            _p_ref   = _p_disk * self.n_rotors / self._SYSTEM_ETA_REF
            _bat_est = (_p_ref * self.min_endurance_min / 60.0
                       ) / (self.battery_energy_density
                             * max(self.battery_efficiency, 1e-6))
            if _bat_est > self.payload_mass:
                warnings.append(
                    f"Estimated battery mass ({_bat_est:.2f} kg) > payload_mass "
                    f"({self.payload_mass:.2f} kg) — the battery alone outweighs "
                    f"the payload. Reduce min_endurance_min or increase "
                    f"battery_energy_density."
                )
            # Disk loading check: thrust per unit disk area.
            # Typical UAVs: 50–200 N/m².  Above ~250 N/m² the design becomes
            # noisy, power-hungry and very sensitive to tip speed limits.
            _disk_loading = _t_rotor / max(_area, 1e-6)   # N/m²
            if _disk_loading > 250:
                warnings.append(
                    f"Disk loading {_disk_loading:.0f} N/m² is very high — "
                    f"typical UAVs are 50–200 N/m². Consider increasing "
                    f"max_diameter or n_rotors."
                )
        if self.min_endurance_min <= 0:
            warnings.append(
                f"min_endurance_min={self.min_endurance_min} must be positive"
            )
        thrust_per_rotor = self.payload_mass * 9.81 / max(1, self.n_rotors)
        if thrust_per_rotor > 500:
            warnings.append(
                f"payload thrust per rotor ({thrust_per_rotor:.1f} N) is in "
                f"high-disk-loading territory — double-check payload_mass/n_rotors"
            )
        if not os.path.exists(self.motor_db_path):
            warnings.append(
                f"Motor database not found at '{self.motor_db_path}' — "
                f"motor selection will fail at runtime"
            )
        else:
            try:
                with open(self.motor_db_path, newline="",
                          encoding="utf-8-sig") as _mf:
                    _entries = [r for r in csv.DictReader(_mf, delimiter=";")
                                if r.get("name", "").strip()]
                if not _entries:
                    warnings.append(
                        f"Motor database '{self.motor_db_path}' contains no valid "
                        f"entries — motor selection will fail at runtime"
                    )
            except Exception as _me:
                warnings.append(f"Could not read motor database: {_me}")
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
        self.min_endurance_min      = 10.0
        self.battery_capacity_Wh    = 0.0
        self.w_power                = 0.5
        self.w_mass                 = 0.0
        self.w_endurance            = 0.5
        self.tip_speed_max          = 200.0
        self.air_density            = 1.225
        self.initial_tip_speed      = 65.0
        self.airfoil_candidates     = ["4412", "2412", "6412", "4409", "4415", "0012", "0009"]
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

    def _build_spanwise_fig(self, figsize=(7, 8)):
        """Build and return the 3-panel spanwise distribution figure, or None if no data."""
        import matplotlib.pyplot as plt
        rows = self.propeller.spanwise_distribution
        if not rows:
            return None
        r     = [row[0] for row in rows]
        chord = [row[1] * 1000 for row in rows]
        pitch = [row[2] for row in rows]
        dT    = [row[3] for row in rows]
        dQ    = [row[4] * 1000 for row in rows]

        prop  = self.propeller
        r_hub = prop.hub_radius
        r_tip = prop.diameter / 2
        r_root           = r_hub + prop.root_cutoff_ratio * (r_tip - r_hub)
        eff_min_chord_mm = max(prop.min_chord, prop.min_chord_fraction * (r_tip - r_hub)) * 1000

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        axes[0].plot(r, chord, marker=".")
        axes[0].set_ylabel("chord [mm]")
        axes[0].set_ylim(bottom=0)
        axes[0].axhline(eff_min_chord_mm, color="gray", linestyle=":", linewidth=1.0,
                        label=f"min chord ({eff_min_chord_mm:.1f} mm)")
        axes[0].axvline(r_root, color="gray", linestyle="--", linewidth=1.0,
                        label=f"aero start (r={r_root:.3f} m)")
        axes[0].legend(fontsize=7, loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(r, pitch, marker=".", color="tab:orange")
        axes[1].set_ylabel("pitch [deg]")
        axes[1].axvline(r_root, color="gray", linestyle="--", linewidth=1.0)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(r, dT, marker=".", label="dT [N]")
        axes[2].plot(r, dQ, marker=".", label="dQ [mNm]", color="tab:red")
        axes[2].axvline(r_root, color="gray", linestyle="--", linewidth=1.0, label="aero start")
        axes[2].set_ylabel("section load")
        axes[2].set_xlabel("radius [m]")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            f"NACA {self.airfoil_type} | {int(self.n_blades)} blades | "
            f"D={self.diameter:.3f} m | RPM={self.rpm:.0f}",
            fontsize=11
        )
        fig.tight_layout()
        return fig

    @action(label="Plot spanwise distribution")
    def plot_spanwise(self):
        """GUI Action: chord, pitch and sectional thrust/torque distributions along the span."""
        import matplotlib.pyplot as plt
        fig = self._build_spanwise_fig(figsize=(7, 8))
        if fig is None:
            print("No spanwise distribution available.")
            return
        plt.show()

    def _build_structural_fig(self, figsize=(7, 8)):
        """Build and return the 3-panel structural analysis figure, or None if no data."""
        import matplotlib.pyplot as plt
        sa = self.propeller.structural_analysis
        if not sa:
            return None
        r              = sa["radii"]
        sig_bend_mpa   = [v / 1e6 for v in sa["sigma_bend"]]
        sig_total_mpa  = [v / 1e6 for v in sa["sigma_total"]]
        fos            = sa["FoS"]
        delta_mm       = [v * 1000 for v in sa["delta"]]
        sig_ult_mpa    = sa["sigma_ult"] / 1e6

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        axes[0].plot(r, sig_bend_mpa,  marker=".", label="Bending")
        axes[0].plot(r, sig_total_mpa, marker=".", label="Total (+ centrifugal)", color="tab:red")
        axes[0].axhline(sig_ult_mpa, color="crimson", linestyle="--", linewidth=1.0,
                        label=f"sigma_ult ({sig_ult_mpa:.0f} MPa)")
        axes[0].set_ylabel("Stress [MPa]")
        axes[0].set_ylim(bottom=0)
        axes[0].legend(fontsize=7, loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(r, fos, marker=".", color="tab:green")
        axes[1].axhline(self.structural_fos_min, color="orange", linestyle="--", linewidth=1.0,
                        label=f"min FoS = {self.structural_fos_min}")
        axes[1].set_ylabel("Factor of Safety [-]")
        axes[1].set_ylim(bottom=0)
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(r, delta_mm, marker=".", color="tab:purple")
        axes[2].set_ylabel("Deflection [mm]")
        axes[2].set_xlabel("radius [m]")
        axes[2].grid(True, alpha=0.3)

        min_fos = sa["min_FoS"]
        r_crit  = sa["critical_radius_m"]
        tip_mm  = sa["tip_deflection_m"] * 1000
        fig.suptitle(
            f"Structural Analysis — NACA {self.airfoil_type} | {self.propeller_material} | "
            f"min FoS={min_fos:.2f} @ r={r_crit * 1000:.0f}mm | tip d={tip_mm:.1f}mm",
            fontsize=10
        )
        fig.tight_layout()
        return fig

    @action(label="Plot structural analysis")
    def plot_structural(self):
        """GUI Action: Euler-Bernoulli stress, factor of safety, and tip deflection vs span."""
        import matplotlib.pyplot as plt
        fig = self._build_structural_fig(figsize=(7, 8))
        if fig is None:
            print("No structural analysis data available.")
            return
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

    def _build_motor_curve_fig(self, motor, name, figsize=(7, 5)):
        """Build and return the motor torque-vs-speed figure."""
        import matplotlib.pyplot as plt
        kv     = motor.kv
        R      = motor.resistance / 1000
        kt     = motor.kt
        V      = motor.voltage_required
        kv_rad = kv * 2.0 * math.pi / 60.0
        rpm_arr   = [self.rpm * i * 0.05 for i in range(1, 41)]
        omega_arr = [r * 2.0 * math.pi / 60.0 for r in rpm_arr]
        torque    = [max(0.0, (V - om / kv_rad) * kt / max(R, 1e-6)) for om in omega_arr]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(rpm_arr, torque, label=f"{name} @ {V:.1f} V")
        ax.axhline(0.8 * motor.max_current * kt, color="red",
                   linestyle="--", label="80% current limit")
        ax.scatter([self.rpm], [motor.torque_req], color="black",
                   zorder=5, label="operating point")
        ax.set_xlabel("RPM")
        ax.set_ylabel("Torque [Nm]")
        ax.set_title(f"Motor: {name}  |  Efficiency: {motor.efficiency:.1%}  |  KV: {kv} RPM/V")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    @action(label="Plot motor curve")
    def plot_motor_curve(self):
        """GUI Action: motor torque-vs-speed curve at the operating voltage."""
        import matplotlib.pyplot as plt
        if not self.feasible_motors:
            print("No feasible motor — nothing to plot.")
            return
        name, motor = self.best_motor
        self._build_motor_curve_fig(motor, name)
        plt.show()

    @action(label="Export design PDF")
    def export_pdf(self):
        """
        GUI Action: write a multi-page PDF report to data/output/design_<timestamp>.pdf.
        Uses matplotlib PdfPages (no extra dependency).

        Pages:
          1 — Mission inputs + optimized outputs summary table
          2 — Mass budget pie chart
          3 — Spanwise chord / pitch / thrust/torque distributions
          4 — Motor torque-vs-speed curve (or "no motor" notice)
          5 — BEMT health summary bar chart
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from datetime import datetime

        out_dir = os.path.join("data", "output")
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path  = os.path.join(out_dir, f"design_{stamp}.pdf")

        perf   = self.propeller.performance
        check  = self.thrust_check
        health = self.propeller.aero_health_summary
        omega  = self.rpm * 2.0 * math.pi / 60.0
        mach_tip = (omega * self.diameter / 2.0) / self.speed_of_sound

        motor_name = "—"
        motor_eff  = "—"
        motor_mass_g = 0.0
        if self.feasible_motors:
            motor_name, _m = self.best_motor
            motor_eff  = f"{_m.efficiency:.1%}"
            motor_mass_g = _m.mass

        with PdfPages(path) as pdf:

            # ── Page 1: Summary table ─────────────────────────────────────────
            fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait
            fig.patch.set_facecolor("white")

            fig.text(0.5, 0.96, "UAV Propulsion System — Design Report",
                     ha="center", va="top", fontsize=16, fontweight="bold")
            fig.text(0.5, 0.93, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                     ha="center", va="top", fontsize=9, color="gray")

            # Left column — mission inputs
            # Split airfoil candidates across two lines if the list is long
            _afs  = self.airfoil_candidates
            _half = (len(_afs) + 1) // 2   # ceiling half — first line slightly longer
            _af_rows = (
                [("Airfoil candidates",
                  ", ".join(str(a) for a in _afs[:_half])),
                 ("",
                  ", ".join(str(a) for a in _afs[_half:]))]
                if len(_afs) > 4
                else [("Airfoil candidates",
                       ", ".join(str(a) for a in _afs))]
            )
            left_items = [
                ("MISSION INPUTS", ""),
                ("Payload mass",          f"{self.payload_mass:.3f} kg"),
                ("Number of rotors",      f"{self.n_rotors}"),
                ("Safety margin",         f"{self.safety_margin:.2f}"),
                ("Max diameter",          f"{self.max_diameter:.3f} m"),
                ("Propeller material",    self.propeller_material),
                ("Battery energy density",f"{self.battery_energy_density:.0f} Wh/kg"),
                ("Battery efficiency",    f"{self.battery_efficiency:.0%}"),
                ("Min endurance",         f"{self.min_endurance_min:.1f} min"),
                ("w_power / w_mass / w_endurance",
                 f"{self.w_power} / {self.w_mass} / {self.w_endurance}"),
                *_af_rows,
                ("Blade candidates",      ", ".join(str(b) for b in self.blade_candidates)),
            ]

            # Right column — optimized outputs
            right_items = [
                ("OPTIMIZED DESIGN", ""),
                ("Airfoil",              f"NACA {self.airfoil_type}"),
                ("Number of blades",     f"{int(self.n_blades)}"),
                ("Diameter",             f"{self.diameter:.3f} m"),
                ("RPM",                  f"{self.rpm:.0f}"),
                ("Thrust / rotor",       f"{perf['thrust']:.2f} N"),
                ("Torque / rotor",       f"{perf['torque']:.3f} Nm"),
                ("Shaft power / rotor",  f"{perf['shaft_power']:.1f} W"),
                ("Tip Mach",             f"{mach_tip:.3f}"),
                ("Thrust status",        check["status"]),
                ("Motor",                motor_name),
                ("Motor efficiency",     motor_eff),
                ("Battery capacity",     f"{self.battery.capacity_Wh:.1f} Wh"),
                ("Endurance",            f"{self.battery.endurance_min:.1f} min"),
                ("Total mass",           f"{self.total_mass * 1000:.0f} g"),
            ]

            y_start = 0.88
            row_h   = 0.035
            x_l_key, x_l_val = 0.04, 0.30
            x_r_key, x_r_val = 0.52, 0.78

            for i, (key, val) in enumerate(left_items):
                y = y_start - i * row_h
                bold = (val == "")
                fw = "bold" if bold else "normal"
                col = "#1a1a6e" if bold else "black"
                fig.text(x_l_key, y, key, fontsize=9, fontweight=fw, color=col, va="top")
                if val:
                    fig.text(x_l_val, y, val, fontsize=9, va="top")

            for i, (key, val) in enumerate(right_items):
                y = y_start - i * row_h
                bold = (val == "")
                fw = "bold" if bold else "normal"
                col = "#1a1a6e" if bold else "black"
                fig.text(x_r_key, y, key, fontsize=9, fontweight=fw, color=col, va="top")
                if val:
                    fig.text(x_r_val, y, val, fontsize=9, va="top")

            # Dividing line
            fig.add_artist(plt.Line2D([0.5, 0.5],
                                      [0.05, y_start + row_h],
                                      transform=fig.transFigure,
                                      color="lightgray", linewidth=0.8))

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── Page 2: Mass budget pie chart ─────────────────────────────────
            fig, ax = plt.subplots(figsize=(8.27, 6))
            rotor_mass_g = self.n_rotors * self.propeller.mass * 1000
            motor_mass_total_g = (self.n_rotors * motor_mass_g
                                  if self.feasible_motors else
                                  self.n_rotors * self.motor_mass_estimate_g)
            bat_mass_g = self.battery.mass * 1000
            payload_g  = self.payload_mass * 1000
            labels  = ["Payload", f"Rotors (×{self.n_rotors})",
                       f"Motors (×{self.n_rotors})", "Battery"]
            sizes   = [payload_g, rotor_mass_g, motor_mass_total_g, bat_mass_g]
            colors  = ["#4c72b0", "#55a868", "#c44e52", "#dd8452"]
            explode = [0.02] * 4
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, explode=explode,
                autopct=lambda p: f"{p:.1f}%\n({p * sum(sizes) / 100:.0f} g)",
                startangle=140, pctdistance=0.75,
            )
            for at in autotexts:
                at.set_fontsize(8)
            ax.set_title(
                f"Mass Budget — Total MTOM: {self.total_mass * 1000:.0f} g",
                fontsize=12, fontweight="bold"
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── Page 3: Spanwise distributions ───────────────────────────────
            fig = self._build_spanwise_fig(figsize=(8.27, 9))
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # ── Page 4: Motor curve ───────────────────────────────────────────
            if self.feasible_motors:
                name, motor = self.best_motor
                fig = self._build_motor_curve_fig(motor, name, figsize=(8.27, 5))
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            else:
                fig = plt.figure(figsize=(8.27, 4))
                fig.text(0.5, 0.5, "No feasible motor selected.\n"
                         "Add higher-rated motors to motors.csv or reduce RPM.",
                         ha="center", va="center", fontsize=12, color="red")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # ── Page 5: BEMT health bar chart ─────────────────────────────────
            total    = health["total_sections"]
            healthy  = health["healthy_count"]
            stalled  = len(health["stalled_radii"])
            diverged = len(health["diverged_radii"])
            non_conv = len(health["non_converged_radii"])
            skipped  = health["skipped_count"]

            fig, ax = plt.subplots(figsize=(8.27, 4))
            categories = ["Healthy", "Stalled", "Diverged", "Non-converged", "Skipped"]
            counts     = [healthy, stalled, diverged, non_conv, skipped]
            bar_colors = ["#55a868", "#c44e52", "#dd8452", "#c49e52", "#aaaaaa"]
            bars = ax.barh(categories, counts, color=bar_colors, edgecolor="white")
            for bar, cnt in zip(bars, counts):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                        str(cnt), va="center", fontsize=9)
            ax.set_xlabel("Section count")
            ax.set_title(
                f"BEMT Health — {healthy}/{total} healthy sections "
                f"({100.0 * healthy / max(total, 1):.0f}%)",
                fontsize=11
            )
            ax.set_xlim(right=max(counts) * 1.15 + 1)
            ax.grid(axis="x", alpha=0.3)

            bad = stalled + diverged + non_conv
            if total > 0 and bad / total > 0.20:
                ax.text(0.5, -0.18,
                        f"WARNING: {bad}/{total} sections unreliable "
                        f"({bad / total:.0%}) — aerodynamic results may be inaccurate",
                        ha="center", va="top", transform=ax.transAxes,
                        fontsize=9, color="red",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0",
                                  edgecolor="red", alpha=0.8))

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── Page 6: Structural analysis ───────────────────────────────
            fig = self._build_structural_fig(figsize=(8.27, 10))
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"PDF report exported to {path}")
        self._show_popup("PDF Exported", f"Report saved to:\n{path}", kind="info")
        return path

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
            capacity_Wh          = self.battery_capacity_Wh,
            power_electrical     = self.total_electrical_power,
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
