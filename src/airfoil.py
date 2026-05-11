import math
import os
import subprocess

import numpy as np
from parapy.core import Base, Input, Attribute, Part
from parapy.geom import BSplineCurve, Position
from scipy.interpolate import interp1d
from .config import get_value


class Airfoil(Base):
    """
    Handles XFOIL integration and identifies optimal operating points.
    Also generates NACA 4-digit coordinates for 3D geometry lofting.
    """

    #: required input slot — NACA 4-digit code as string
    naca_code = Input(get_value("airfoil", "naca_code", default="4412"))

    #: optional input slot — Reynolds number for viscous polar
    reynolds = Input(get_value("airfoil", "reynolds", default=300000))

    #: optional input slot — number of points for geometry generation
    n_points = Input(get_value("airfoil", "n_points", default=60))

    #: optional input slot — chord length for geometry scaling
    chord = Input(get_value("airfoil", "chord", default=1.0))

    #: optional input slot — thickness scale factor
    thickness_factor = Input(get_value("airfoil", "thickness_factor", default=1.0))

    #: optional input slot — placement in 3D space
    position = Input(Position())

    @Attribute
    def polar_data(self):
        """
        Integration Rule: runs XFOIL via subprocess, parses the polar
        file and builds interpolators for Cl and Cd.
        """
        polar_file = f"polar_{self.naca_code}_{int(self.reynolds)}.txt"

        # Check if polar file already exists
        if not os.path.exists(polar_file):
            commands = (
                f"NACA {self.naca_code}\n"
                "PANE\n"
                "OPER\n"
                f"VISC {self.reynolds}\n"
                "ITER 200\n"
                "PACC\n"
                f"{polar_file}\n\n"
                f"ASEQ -5 20 0.5\n"
                "PACC\n"
                "QUIT\n"
            )

            process = subprocess.Popen(
                "xfoil",
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            process.communicate(commands)

        # Parse polar file
        raw_data = []
        if os.path.exists(polar_file):
            with open(polar_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            raw_data.append((
                                float(parts[0]),
                                float(parts[1]),
                                float(parts[2])
                            ))
                        except ValueError:
                            continue

        if not raw_data:
            raise RuntimeError(
                f"XFOIL failed for NACA {self.naca_code} at Re={self.reynolds}. "
                f"Possible causes:\n"
                f"  1. XFOIL is not on your system PATH — type 'xfoil' in terminal to verify\n"
                f"  2. NACA code '{self.naca_code}' is invalid — must be 4 digits\n"
                f"  3. Reynolds number {self.reynolds} is too low for convergence\n"
                f"  4. Alpha sweep -5 to 20 deg produced no converged solutions"
            )

        # Logic Rule: remove duplicate alpha entries, keep last occurrence
        unique_points = {
            item[0]: (item[1], item[2])
            for item in sorted(raw_data)
        }

        alphas = sorted(unique_points.keys())
        cls    = [unique_points[a][0] for a in alphas]
        cds    = [unique_points[a][1] for a in alphas]

        # Mathematical Rule: find optimal L/D operating point
        l_over_d = [cl / max(cd, 1e-6) for cl, cd in zip(cls, cds)]
        idx = int(np.argmax(l_over_d))

        # Scipy Integration: build interpolators for BEM solver
        cl_interp = interp1d(alphas, cls, kind='linear',
                             fill_value="extrapolate")
        cd_interp = interp1d(alphas, cds, kind='linear',
                             fill_value="extrapolate")

        return {
            "cl_interp"     : cl_interp,
            "cd_interp"     : cd_interp,
            "alpha_opt_rad" : math.radians(alphas[idx]),
            "cl_opt"        : cls[idx]
        }

    def get_cl_cd(self, alpha_rad):
        """Evaluates the polar interpolators at a given angle of attack."""
        data    = self.polar_data
        alpha_deg = math.degrees(alpha_rad)
        return (
            float(data["cl_interp"](alpha_deg)),
            float(data["cd_interp"](alpha_deg))
        )

    @Part(parse=False)
    def curve(self):
        points = self._clean_points(self.points)
        if len(points) < 4:
            points = self._clean_points(self._fallback_points())
        return BSplineCurve(control_points=points, degree=3)

    @Attribute
    def points(self):
        file_path = os.path.join("data", "airfoil_library", f"naca{self.naca_code}.dat")
        debug = bool(get_value("debug", "airfoil", default=False))

        # DEBUG 1: Verify path
        if debug:
            print(f"\n--- DEBUG: Point Generation for NACA {self.naca_code} ---")
            print(f"Target Path: {os.path.abspath(file_path)}")

        if not os.path.exists(file_path):
            if debug:
                print(f"CRITICAL ERROR: File not found at {file_path}")
            return self._fallback_points()

        point_lst = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if debug:
                    print(f"Total lines in file: {len(lines)}")

                # Check the first few lines to ensure it's not binary or empty
                if len(lines) > 0:
                    if debug:
                        print(f"First line (header): {lines[0].strip()}")

                for i, line in enumerate(lines[1:]):  # Skip header
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            z = float(parts[1])

                            # Scaling logic
                            pt = self.position.translate(
                                "x", x * self.chord,
                                "z", z * self.chord * self.thickness_factor
                            )
                            point_lst.append(pt)
                        except ValueError:
                            if debug:
                                print(f"Line {i + 2} skipped (invalid floats): {line.strip()}")

            if debug:
                print(f"Successfully generated {len(point_lst)} points.")

        except Exception as e:
            if debug:
                print(f"Unexpected error during file read: {e}")

        # DEBUG 2: Final Validation
        if len(point_lst) < 2:
            if debug:
                print("ERROR: Resulting point list is too short for a curve!")
            # This is why you get the Geom_Curve error. We need at least 2 points.
            return self._fallback_points()

        return point_lst

    def _clean_points(self, points):
        """Remove duplicate/degenerate points to keep the curve builder stable."""
        cleaned = []
        last = None
        tol = 1e-8
        for pt in points:
            if last is None:
                cleaned.append(pt)
                last = pt
                continue

            if (pt.x - last.x) ** 2 + (pt.y - last.y) ** 2 + (pt.z - last.z) ** 2 > tol:
                cleaned.append(pt)
                last = pt

        if len(cleaned) < 4:
            return self._fallback_points()

        max_pts = 16
        if len(cleaned) > max_pts:
            step = max(1, len(cleaned) // (max_pts - 1))
            sampled = cleaned[::step]
            if sampled[-1] is not cleaned[-1]:
                sampled.append(cleaned[-1])
            cleaned = sampled

        return cleaned


    def _fallback_points(self):
        """Generate a simple symmetric profile if no .dat file is available."""
        code = "".join(ch for ch in str(self.naca_code) if ch.isdigit())
        try:
            thickness_pct = int(code[-2:]) if len(code) >= 2 else 12
        except ValueError:
            thickness_pct = 12

        t = max(1.0, min(40.0, thickness_pct)) / 100.0
        x_vals = np.linspace(0.0, 1.0, self.n_points)
        y_t = 5.0 * t * (
            0.2969 * np.sqrt(x_vals)
            - 0.1260 * x_vals
            - 0.3516 * x_vals ** 2
            + 0.2843 * x_vals ** 3
            - 0.1015 * x_vals ** 4
        )

        upper = [self.position.translate("x", x * self.chord, "z", y * self.chord * self.thickness_factor)
                 for x, y in zip(reversed(x_vals), reversed(y_t))]
        lower = [self.position.translate("x", x * self.chord, "z", -y * self.chord * self.thickness_factor)
                 for x, y in zip(x_vals, y_t)]

        return upper + lower[1:]