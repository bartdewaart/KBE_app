import math
import os
import subprocess

import numpy as np
from parapy.core import Base, Input, Attribute
from scipy.interpolate import interp1d


class Airfoil(Base):
    """
    Handles XFOIL integration and identifies optimal operating points.
    Also generates NACA 4-digit coordinates for 3D geometry lofting.
    """

    #: required input slot — NACA 4-digit code as string
    naca_code = Input("4412")

    #: optional input slot — Reynolds number for viscous polar
    reynolds = Input(300000)

    #: optional input slot — number of points for geometry generation
    n_points = Input(60)

    @Attribute
    def polar_data(self):
        """
        Integration Rule: runs XFOIL via subprocess, parses the polar
        file and builds interpolators for Cl and Cd.

        NOTE: deleting the stale polar file before running is an
        intentional side effect — without it XFOIL appends to old data
        and produces corrupt polars. This is the only side effect in
        this attribute and is unavoidable given XFOIL's file-based I/O.
        """
        polars_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "polars")
        os.makedirs(polars_dir, exist_ok=True)
        polar_file = os.path.join(polars_dir, f"polar_{self.naca_code}_{int(self.reynolds)}.txt")

        polar_filename = os.path.basename(polar_file)

        if not os.path.exists(polar_file):
            commands = (
                f"NACA {self.naca_code}\n"
                "PANE\n"
                "OPER\n"
                f"VISC {self.reynolds}\n"
                "ITER 200\n"
                "PACC\n"
                f"{polar_filename}\n\n"
                f"ASEQ -5 20 0.5\n"
                "PACC\n"
                "QUIT\n"
            )

            process = subprocess.Popen(
                "xfoil",
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=polars_dir,
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

        # Mathematical Rule: find optimal L/D operating point.
        # Restrict search to the pre-stall regime (alpha <= alpha at CL_max).
        # XFOIL can produce anomalously low Cd near/post stall due to
        # convergence failure, which would otherwise cause argmax to select
        # a stalled alpha as "optimal" — leading to over-pitched Betz blades.
        idx_clmax = int(np.argmax(cls))
        l_over_d = []
        for i, (cl, cd) in enumerate(zip(cls, cds)):
            if i <= idx_clmax and cl > 0 and cd > 0:
                l_over_d.append(cl / cd)
            else:
                l_over_d.append(-1.0)
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
            "cl_opt"        : cls[idx],
            "alpha_min_deg" : alphas[0],
            "alpha_max_deg" : alphas[-1]
        }

    def get_cl_cd(self, alpha_rad):
        """Evaluates the polar interpolators at a given angle of attack."""
        data    = self.polar_data
        alpha_deg = math.degrees(alpha_rad)
        return (
            float(data["cl_interp"](alpha_deg)),
            float(data["cd_interp"](alpha_deg))
        )

    @Attribute
    def points(self):
        """
        Geometry Rule: generates NACA 4 or 5-digit airfoil coordinates
        for the FittedCurve in BladeSection.
        Returns a counter-clockwise list of [x, y, 0] points.
        """
        if len(self.naca_code) not in (4, 5):
            raise ValueError(
                f"NACA code '{self.naca_code}' is not a valid 4 or 5-digit code. "
                f"Example valid codes: '4412', '23012'."
            )

        x = np.linspace(0, 1, self.n_points)

        # Thickness distribution (same formula for both 4 and 5-digit series)
        t = int(self.naca_code[-2:]) / 100.0
        yt = (5 * t * (
                0.2969 * np.sqrt(x)
                - 0.1260 * x
                - 0.3516 * x ** 2
                + 0.2843 * x ** 3
                - 0.1015 * x ** 4
        ))

        if len(self.naca_code) == 4:
            # ── NACA 4-digit camber line ──────────────────────────────────
            # Parameters extracted from code digits:
            # digit 1 → max camber (m), digit 2 → camber position (p)
            m = int(self.naca_code[0]) / 100.0
            p = int(self.naca_code[1]) / 10.0

            # Domain Protection: avoid division by zero for symmetric
            # airfoils where p == 0 (e.g. NACA 0012)
            if p < 1e-6:
                xcam = np.zeros_like(x)
            else:
                xcam = np.where(
                    x < p,
                    m / p ** 2 * (2 * p * x - x ** 2),
                    m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2)
                )

        else:
            # ── NACA 5-digit camber line ──────────────────────────────────
            # Parameters encoded differently from 4-digit series:
            # digit 1 → design lift coefficient = digit * 3/20
            # digit 2 → camber type: 1 = standard, 2 = reflexed
            # digit 3 → max camber position p = digit / 20
            # digits 4-5 → thickness (same as 4-digit)
            reflex = int(self.naca_code[1])
            p = int(self.naca_code[2]) / 20.0

            # Logic Rule: reflexed camber line requires additional constants
            # and a different formula — not yet implemented
            if reflex == 2:
                raise NotImplementedError(
                    f"NACA 5-digit reflexed camber line (second digit = 2) "
                    f"is not yet implemented. '{self.naca_code}' was provided."
                )

            # Mathematical Rule: k1 constants are tabulated from the original
            # NACA report for each standard camber position value
            k1_table = {
                0.05: 361.4,
                0.10: 51.64,
                0.15: 15.957,
                0.20: 6.643,
                0.25: 3.230,
            }

            # Domain Protection: p must correspond to a standard tabulated value
            # Valid third digits are 1-5 giving p = 0.05 to 0.25
            if round(p, 2) not in k1_table:
                raise ValueError(
                    f"NACA 5-digit code '{self.naca_code}' has p={p:.2f} "
                    f"which is not a standard value. "
                    f"Valid third digits are 1-5 (p = 0.05 to 0.25)."
                )

            k1 = k1_table[round(p, 2)]

            # Mathematical Rule: standard non-reflexed 5-digit camber line
            # forward of max camber position uses cubic polynomial,
            # aft section is linear decay to trailing edge
            xcam = np.where(
                x < p,
                (k1 / 6.0) * (x ** 3
                              - 3 * p * x ** 2
                              + p ** 2 * (3 - p) * x),
                (k1 * p ** 3 / 6.0) * (1 - x)
            )

        pts_upper = [[xi, yi + yti, 0]
                     for xi, yi, yti in zip(x, xcam, yt)]
        pts_lower = [[xi, yi - yti, 0]
                     for xi, yi, yti in zip(x, xcam, yt)]

        # Counter-clockwise loop: upper surface TE→LE, lower surface LE→TE
        return pts_upper[::-1] + pts_lower[1:]