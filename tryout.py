import math
import os
import subprocess
import numpy as np
from parapy.core import Base, Input, Attribute, Part, Sequence
from scipy.interpolate import interp1d

# --- 1. AERODYNAMIC DATA LAYER ---
class Airfoil(Base):
    """Handles XFOIL integration to identify optimal operating points."""
    naca_code = Input("4412")
    reynolds = Input(300000)

    @Attribute
    def polar_data(self):
        """Runs XFOIL and identifies alpha_opt and cl_opt for the sizing logic."""
        polar_file = f"polar_{self.naca_code}.txt"
        if os.path.exists(polar_file):
            os.remove(polar_file)

        commands = (f"NACA {self.naca_code}\nPANE\nOPER\nVISC {self.reynolds}\n"
                    f"PACC\n{polar_file}\n\nASEQ -5 15 0.5\nQUIT\n")

        process = subprocess.Popen("xfoil", stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, text=True)
        process.communicate(commands)

        raw_data = []
        if os.path.exists(polar_file):
            with open(polar_file, 'r') as f:
                lines = f.readlines()
                for line in lines[12:]:
                    p = line.split()
                    if len(p) >= 3:
                        raw_data.append([float(p[0]), float(p[1]), float(p[2])])
        
        if not raw_data:
            raise RuntimeError("XFOIL failed to generate polar data. Check your XFOIL path.")

        data_np = np.array(raw_data)
        alphas, cls, cds = data_np[:, 0], data_np[:, 1], data_np[:, 2]
        l_over_d = cls / np.maximum(cds, 1e-6)
        best_idx = np.argmax(l_over_d)

        return {
            "alpha_opt_rad": math.radians(alphas[best_idx]),
            "cl_opt": cls[best_idx],
            "cl_interp": interp1d(alphas, cls, fill_value="extrapolate"),
            "cd_interp": interp1d(alphas, cds, fill_value="extrapolate")
        }

    def get_cl_cd(self, alpha_rad):
        """Evaluates airfoil coefficients from the generated polar interpolators."""
        alpha_deg = math.degrees(alpha_rad)
        cl = float(self.polar_data["cl_interp"](alpha_deg))
        cd = float(self.polar_data["cd_interp"](alpha_deg))
        return cl, cd

# --- 2. 2D PHYSICS LAYER ---
class BladeSection(Base):
    """
    Computes local aerodynamic loads using the iterative BEMT solver.
    Navigates the model tree to fetch global parameters and sizing logic.
    """

    @Attribute
    def prop_ref(self):
        """Tree Traversal: Section -> Section Sequence -> Blade -> Blade Sequence -> Propeller."""
        return self.parent.parent

    @Attribute
    def radius(self):
        """Geometry Rule: Placement based on hub radius and spanwise step size."""
        p = self.prop_ref
        dr = (p.diameter / 2 - 0.02) / p.n_segments
        return 0.02 + (self.index + 0.5) * dr

    @Attribute
    def geometry(self):
        """Retrieves optimal chord and pitch from the generative sizing logic."""
        return self.parent.sizing_logic(self.radius)

    @Attribute
    def aerodynamics(self):
        """Analysis Rule: Iterative BEMT solver for sectional thrust and torque."""
        p = self.prop_ref
        chord, pitch = self.geometry
        omega = p.rpm * 2.0 * math.pi / 60.0
        rho = 1.225
        
        # Initial guess from 1D momentum theory
        v_i = math.sqrt(p.target_thrust / (2.0 * rho * (math.pi * (p.diameter / 2)**2)))
        v_theta = 0.0
        relaxation = 0.1
        dr = (p.diameter / 2 - 0.02) / p.n_segments

        dT, dQ = 0.0, 0.0
        for _ in range(100):
            v_ax = v_i
            v_rot = omega * self.radius - v_theta
            v_eff = math.sqrt(v_ax**2 + v_rot**2)
            phi = math.atan2(v_ax, v_rot)
            alpha = pitch - phi
            
            cl, cd = p.airfoil.get_cl_cd(alpha)
            
            # Mathematical Rules: Sectional Aerodynamic Forces
            L_p = 0.5 * rho * v_eff**2 * chord * cl
            D_p = 0.5 * rho * v_eff**2 * chord * cd
            
            # Loads per unit span for a single blade
            dT = (L_p * math.cos(phi) - D_p * math.sin(phi)) * dr
            dQ = (L_p * math.sin(phi) + D_p * math.cos(phi)) * self.radius * dr

            # Prandtl Tip-Loss Correction
            f_tip = (p.n_blades / 2) * (p.diameter / 2 - self.radius) / (self.radius * max(1e-6, math.sin(phi)))
            F = (2 / math.pi) * math.acos(max(0, min(1, math.exp(-f_tip))))
            F_iter = max(1e-6, F)

            # Momentum Balance Rule
            v_i_new = math.sqrt(abs(dT * p.n_blades) / (4 * math.pi * self.radius * rho * F_iter * dr))
            
            if abs(v_i_new - v_i) < 1e-4: 
                break
            v_i = (1 - relaxation) * v_i + relaxation * v_i_new
            v_theta = (dQ * p.n_blades) / (4 * math.pi * self.radius**2 * rho * v_i_new * F_iter * dr) if v_i_new > 0 else 0

        return {"dT": dT, "dQ": dQ}

# --- 3. GEOMETRY & ASSEMBLY LAYER ---
class Blade(Base):
    """Generates the optimal blade planform and discretizes it into analysis sections."""

    @Attribute
    def sizing_logic(self):
        """Generative Rule: Analytical sizing using Betz optimality conditions."""
        p = self.parent # Blade -> Blade Sequence -> Propeller
        r_tip = p.diameter / 2
        v_i_ideal = math.sqrt(p.target_thrust / (2.0 * 1.225 * (math.pi * r_tip**2)))
        omega = p.rpm * 2.0 * math.pi / 60.0
        cl_opt = p.airfoil.polar_data["cl_opt"]
        alpha_opt = p.airfoil.polar_data["alpha_opt_rad"]

        def calculate_at(r):
            phi = math.atan2(v_i_ideal, omega * r)
            v_eff = math.sqrt(v_i_ideal**2 + (omega * r)**2)
            f_tip = (p.n_blades / 2) * (r_tip - r) / (r * max(1e-6, math.sin(phi)))
            F = (2 / math.pi) * math.acos(max(0, min(1, math.exp(-f_tip))))
            
            # Optimum Chord Generation Rule
            chord = (8 * math.pi * r * v_i_ideal**2 * max(0.1, F)) / (p.n_blades * v_eff**2 * cl_opt * math.cos(phi))
            # Optimum Pitch Rule
            pitch = phi + alpha_opt
            return max(0.01, chord), pitch

        return calculate_at

    @Part(parse=False)
    def sections(self):
        """Configuration Rule: Instantiates the radial analysis segments."""
        return Sequence(type=BladeSection, quantify=self.parent.n_segments)

class Propeller(Base):
    """Root assembly class managing the airfoil, multiple blades, and performance aggregation."""
    diameter = Input(0.4)
    rpm = Input(5000)
    n_blades = Input(2)
    target_thrust = Input(7.36)
    n_segments = Input(20)

    @Part
    def airfoil(self):
        return Airfoil(naca_code="4412")

    @Part
    def blades(self):
        """Instantiates the specified number of blade objects."""
        return Sequence(type=Blade, quantify=self.n_blades)

    @Attribute
    def performance(self):
        """Aggregates total thrust and torque from all spanwise sections across all blades."""
        total_thrust = sum(s.aerodynamics["dT"] for b in self.blades for s in b.sections)
        total_torque = sum(s.aerodynamics["dQ"] for b in self.blades for s in b.sections)
        power = total_torque * (self.rpm * 2.0 * math.pi / 60.0)
        return {"thrust": total_thrust, "torque": total_torque, "power": power}

if __name__ == "__main__":
    # Test case execution
    prop = Propeller()
    perf = prop.performance
    print(f"--- BEMT Propeller Design Summary ---")
    print(f"Target Thrust:   {prop.target_thrust:.2f} N")
    print(f"Analyzed Thrust: {perf['thrust']:.2f} N")
    print(f"Power Required:  {perf['power']:.2f} W")
    print(f"Performance Ratio: {perf['thrust'] / perf['power']:.4f} N/W")