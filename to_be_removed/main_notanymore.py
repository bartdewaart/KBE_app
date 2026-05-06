#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 ParaPy Holding B.V.
#
# This file is subject to the terms and conditions defined in
# the license agreement that you have received with this source code
#
# THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
# PURPOSE.

from parapy.core import Base, Input, Attribute, Part, child


class Wing(Base):
    """
    Creates a parametric wing object
    """
    thickness:float = Input(0.1)   # representative default for the enormously simplified modelling in this class
    chord_root: float = Input()
    chord_tip: float = Input()
    span:float = Input()

    @Attribute
    def area(self) -> float:
        """
        Calculates the area of the wing.
        """
        return (self.chord_tip + self.chord_root) / 2 * self.span

    @Attribute
    def taper_ratio(self) -> float:
        """
        Calculates the taper ratio of the wing
        """
        return self.chord_tip / self.chord_root

    @Attribute
    def aspect_ratio(self) -> float:
        """
        Calculates the aspect ratio of the wing
        """
        return self.span ** 2 / self.area


    @Part
    def airfoil(self):
        # mean airfoil, using the mean chord length
        return Airfoil(thickness=self.thickness,
                       chord=(self.chord_root + self.chord_tip) / 2,
                       hidden=self.chord >= 3,
                       suppress=self.chord <= 0.001 * self.span)


# Now, let's experiment using multiple parts!

class Wing_multi(Wing):
    # default: 2 airfoils (root/chord)
    n_airfoils: int = Input(2)

    @Part
    def airfoil(self):
        return Airfoil(quantify=self.n_airfoils,
                       pass_down="thickness",
                       chord = self.chord_root
                               + child.index
                               * (self.chord_tip - self.chord_root)
                               / (self.n_airfoils - 1),
                       label='root_airfoil' if child.index == 0
                                            else "other_airfoil")


# this class is representing airfoils, although it's not actually doing
# anything useful in this example, except storing chord length and rel. thickness.
# In real applications, you'd probably want it to also provide and airfoil drag
# polar, geometry or some mechanical properties.
class Airfoil(Base):
    thickness = Input()
    chord = Input()


if __name__ == '__main__':
    from parapy.gui import display

    # Single Part
    wing_1af = Wing(chord=1.5, span=15.0, label='simple wing')
    print(wing_1af.airfoil.thickness)
    print(wing_1af.airfoil.chord)
    print(wing_1af.airfoil.hidden)
    wing_1af.chord = 3.5
    print(wing_1af.airfoil.hidden)
    wing_1af.chord = 1e-4
    print(wing_1af.airfoil)

    # Quantified Part
    wing_multiaf = Wing_multi(chord_root=2.5, chord_tip=1.5, span=15.0, label='multi-airfoil wing')
    print(wing_multiaf.airfoils)
    print(len(wing_multiaf.airfoils))
    print(wing_multiaf.airfoils[0].chord)
    print(wing_multiaf.airfoils[-1].chord)
    print([item.chord for item in wing_multiaf.airfoils])

    display([wing_1af, wing_multiaf])

    # Once you run it, check the values of inputs and attributes.
    # Change the inputs and check how it affects the attributes!