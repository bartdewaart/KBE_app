#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import pi

from motor_matching import Motor
from parapy.geom import GeomBase, Box, Cylinder, translate, rotate
from parapy.core import Base, Input, Attribute, Part, child

class PropulsionSystem(Base):
    @Part
    def Motor(self):
        return Motor(KV = 3)

class StairCase(GeomBase):
    """StairCase assembles ``n_step`` steps. Both dimensions and color of
    individual steps are parameterized. Rules for step ``color`` and
    ``position`` rules demonstrate ``child.index`` notation.
    """

    n_step: int = Input()
    w_step: float = Input()
    l_step: float = Input()
    h_step: float = Input() # Att.: This input is not used! How to inlcude it?
    t_step: float = Input()
    colors: list[str] = Input(["red", "green", "blue", "yellow", "orange"])


    @Part
    def steps(self):
        """Translation in a sequence"""
        return Box(quantify=self.n_step,
                   centered=True,
                   width=self.w_step,
                   length=self.l_step,
                   height=self.t_step,
                   color=self.colors[child.index % len(self.colors)],
                   position=self.position if child.index==0 else
                       child.previous.position.translate('y', self.l_step,
                                                         'z', self.h_step)
                   )


class SpiralStairCase(StairCase):
    """Spiraling version of the basic StairCase. Positioning of steps now
    requires additional rotation. Moreover, the staircase assembles extra
    inner and outer columns. Outer column has custom display settings.
    """

    radius: float = Input()
    n_revol: float = Input()
    delta_w: float = Input(0)

    @Attribute
    def angle_step(self):
        """Angle over which each next step should rotate w.r.t. former."""
        return self.n_revol * 2 * pi / (self.n_step - 1)

    @Part
    def steps(self):
        """Translation and rotation in a sequence"""
        return Box(quantify=self.n_step,
                   width=self.w_step,
                   length=self.l_step,
                   height=self.t_step,
                   color=self.colors[child.index % len(self.colors)],
                   position=translate(rotate(self.position,
                                             'z',
                                             child.index*self.angle_step
                                             ),
                                      'x',
                                      self.radius,
                                      'z',
                                      child.index * self.h_step,
                                      '-y',             # This last part is not specified in the exercise, but
                                      self.l_step/2))   # makes the center of the step tangential with the
                                                        # column, which looks better.

    @Part
    def inner_column(self):
        return Cylinder(radius=self.radius,
                        height=self.h_step * self.n_step,
                        color="black")

    @Part
    def outer_column(self):
        return Cylinder(radius=self.radius + self.w_step,
                        height=self.inner_column.height,
                        color="black",
                        display_mode="wireframe",
                        isos=(20, 5))


if __name__ == '__main__':
    from parapy.gui import display

    str_stairs = StairCase(n_step=20, w_step=1.5, l_step=2.0, h_step=0.20,
                           t_step=0.18, label='straight_stairs')
    sp_stairs = SpiralStairCase(n_step=30, w_step=1.5, l_step=1.0, h_step=0.25,
                          t_step=0.18, radius=1, n_revol=2, delta_w=0.1,
                          label='spiral_stairs')
    display([str_stairs, sp_stairs])