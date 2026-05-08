import math
from parapy.core import Base, Input, Attribute
from parapy.geom import FittedCurve
import os
import subprocess
import numpy as np
from scipy.interpolate import interp1d

class Airfoil(FittedCurve):
    """Handles XFOIL integration and identifies optimal operating points."""
    naca_code = Input("4412")
    reynolds = Input()
    chord = Input()

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
    def points(self):  # required slot for FittedCurve superclass
        airfoil_name = str('naca' + self.naca_code)
        if airfoil_name.endswith('.dat'):  # check whether the airfoil name string includes .dat already
            airfoil_file = airfoil_name
        else:
            airfoil_file = airfoil_name + '.dat'
        file_path = os.path.join("data", "airfoil_library", airfoil_file)
        with open(file_path, 'r') as f:
            point_lst = []
            next(f)
            for line in f:
                x, z = line.split(' ', 1)  # the cartesian coordinates are directly interpreted as X and Z coordinates
                point_lst.append(self.position.translate(
                    "x", float(x) * self.chord,  # the x points are scaled according to the airfoil chord length
                    "z", float(
                        z) * self.chord))  # y points are scaled
        return point_lst
