import math
from parapy.core import Base, Input, Attribute


class BladeSection(Base):
    air_density = Input(1.225)

    @Attribute
    def radius(self):
        prop = self.parent
        dr = (prop.diameter / 2 - 0.02) / prop.n_segments
        return 0.02 + (self.index + 0.5) * dr

    @Attribute
    def dr(self):
        prop = self.parent
        return (prop.diameter / 2 - 0.02) / prop.n_segments

    @Attribute
    def chord(self):
        prop = self.parent
        c_spline, _ = prop.splines
        return float(c_spline(self.radius))

    @Attribute
    def pitch(self):
        prop = self.parent
        _, p_spline = prop.splines
        return float(p_spline(self.radius))

    @Attribute
    def total_radius(self):
        return self.parent.diameter / 2

    @Attribute
    def n_blades(self):
        return self.parent.n_blades

    @Attribute
    def rpm(self):
        return self.parent.rpm

    @Attribute
    def airfoil_obj(self):
        return self.parent.airfoil



    @Attribute
    def aerodynamics(self):
        """Calculates local thrust (dT) and torque (dQ)."""
        omega = self.rpm * 2.0 * math.pi / 60.0
        v_ax = math.sqrt(10.0 / (2.0 * self.air_density * math.pi * self.total_radius ** 2))  # Seed guess
        v_rot = omega * self.radius
        phi = math.atan2(v_ax, v_rot)
        v_eff = math.sqrt(v_ax ** 2 + v_rot ** 2)

        cl, cd = self.airfoil_obj.get_cl_cd(self.pitch - phi)

        l_prime = 0.5 * self.air_density * v_eff ** 2 * self.chord * cl
        d_prime = 0.5 * self.air_density * v_eff ** 2 * self.chord * cd

        dT = (l_prime * math.cos(phi) - d_prime * math.sin(phi)) * self.n_blades * self.dr
        dQ = (l_prime * math.sin(phi) + d_prime * math.cos(phi)) * self.radius * self.n_blades * self.dr
        return {"dT": dT, "dQ": dQ}