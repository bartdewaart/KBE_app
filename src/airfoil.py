import math
from parapy.core import Base, Input, Attribute
import os
import subprocess
import numpy as np
from scipy.interpolate import interp1d

class Airfoil(Base):
    """Handles XFOIL integration and identifies optimal operating points."""
    naca_code = Input("4412")
    reynolds = Input()
    n_points = 60

    @Attribute
    def polar_data(self):
        """Runs XFOIL via subprocess and parses results safely."""
        polar_file = f"polar_{self.naca_code}_{int(self.reynolds)}.txt"
        if os.path.exists(polar_file): os.remove(polar_file)

        commands = (f"NACA {self.naca_code}\nPANE\nOPER\nVISC {self.reynolds}\n"
                    f"ITER 200\nPACC\n{polar_file}\n\n"
                    f"ASEQ -5 20 0.5\nPACC\nQUIT\n")

        process = subprocess.Popen("xfoil", stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        process.communicate(commands)

        raw_data = []
        if os.path.exists(polar_file):
            with open(polar_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            raw_data.append((float(parts[0]), float(parts[1]), float(parts[2])))
                        except ValueError: continue

        if not raw_data: raise RuntimeError(f"XFOIL failed for NACA {self.naca_code}")

        # Clean duplicates and find optimal L/D
        unique_points = {item[0]: (item[1], item[2]) for item in sorted(raw_data)}
        alphas = sorted(unique_points.keys())
        cls = [unique_points[a][0] for a in alphas]
        cds = [unique_points[a][1] for a in alphas]

        l_over_d = [cl / max(cd, 1e-6) for cl, cd in zip(cls, cds)]
        idx = np.argmax(l_over_d)

        return {
            "cl_interp": interp1d(alphas, cls, fill_value="extrapolate"),
            "cd_interp": interp1d(alphas, cds, fill_value="extrapolate"),
            "alpha_opt_rad": math.radians(alphas[idx]),
            "cl_opt": cls[idx]
        }

    def get_cl_cd(self, alpha_rad):
        data = self.polar_data
        deg = math.degrees(alpha_rad)
        return float(data["cl_interp"](deg)), float(data["cd_interp"](deg))

    @Attribute
    def points(self):
        """Generates NACA 4-digit coordinates for the FittedCurve."""
        # Simplified NACA 4-digit generator for geometry purposes
        m = int(self.naca_code[0]) / 100.0
        p = int(self.naca_code[1]) / 10.0
        t = int(self.naca_code[2:]) / 100.0

        x = np.linspace(0, 1, self.n_points)
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)

        xcam = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))

        # Upper and Lower points
        pts_upper = [[xi, yi + yti, 0] for xi, yi, yti in zip(x, xcam, yt)]
        pts_lower = [[xi, yi - yti, 0] for xi, yi, yti in zip(x, xcam, yt)]

        return pts_upper[::-1] + pts_lower[1:]  # Counter-clockwise loop