import math
from parapy.core import Base, Input, Attribute


class BladeSection(Base):
    """The core BEMT solver for a single radial station"""

    radius        = Input()
    dr            = Input()
    chord         = Input()
    pitch         = Input()
    total_radius  = Input()
    n_blades      = Input()
    rpm           = Input()
    air_density   = Input(1.225)
    target_thrust = Input()
    airfoil       = Input()

    @Attribute
    def disk_area(self):
        return math.pi * self.total_radius ** 2

    @Attribute
    def aerodynamics(self):
        """Iteratively solves for local thrust (dT) and torque (dQ)."""
        v_i = math.sqrt(self.target_thrust / (2.0 * self.air_density * self.disk_area))
        v_theta = 0.0
        relaxation = 0.1
        tolerance  = 1e-5
        n_iter     = 500
        dT, dQ     = 0.0, 0.0

        for _ in range(n_iter):
            v_ax  = v_i
            v_rot = (self.rpm * 2 * math.pi / 60 * self.radius) - v_theta
            v_eff = math.sqrt(v_ax ** 2 + v_rot ** 2)
            phi   = math.atan2(v_ax, v_rot)
            alpha = self.pitch - phi

            cl, cd  = self.airfoil.get_cl_cd(alpha)   # ← renamed
            l_prime = 0.5 * self.air_density * v_eff ** 2 * self.chord * cl
            d_prime = 0.5 * self.air_density * v_eff ** 2 * self.chord * cd

            dT = (l_prime * math.cos(phi) - d_prime * math.sin(phi)) * self.n_blades * self.dr
            dQ = (l_prime * math.sin(phi) + d_prime * math.cos(phi)) * self.radius * self.n_blades * self.dr

            sin_phi = math.sin(phi)
            if sin_phi <= 0:
                F = 1e-6
            else:
                f_tip = (self.n_blades / 2.0) * (self.total_radius - self.radius) / (self.radius * sin_phi)
                F = (2.0 / math.pi) * math.acos(max(0.0, min(1.0, math.exp(-f_tip))))

            F = max(F, 1e-6)

            v_i_new = math.sqrt(abs(dT) / (4.0 * math.pi * self.radius * self.air_density * F * self.dr))
            if v_i_new > 0:
                v_theta_new = dQ / (4.0 * math.pi * self.radius ** 2 * self.air_density * v_i_new * F * self.dr)
            else:
                v_theta_new = 0.0

            if abs(v_i_new - v_i) < tolerance:
                break

            v_i     = (1 - relaxation) * v_i     + relaxation * v_i_new
            v_theta = (1 - relaxation) * v_theta + relaxation * v_theta_new

        return {"dT": dT, "dQ": dQ}