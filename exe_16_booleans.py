#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2018 ParaPy Holding B.V.
#
# This file is subject to the terms and conditions defined in
# the license agreement that you have received with this source code
#
# THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
# PURPOSE.

from parapy.core import Input, Part
from parapy.geom import GeomBase, Box, Cone, translate, FusedSolid,\
                        SubtractedSolid, CommonSolid, PartitionedSolid,\
                        Plane, VZ, Vector, HalfSpaceSolid


class BuildingSolids(GeomBase):
    # ------------- elementary shapes

    height: float = Input()
    length: float = Input()
    width: float = Input()

    cone_height: float = Input()
    base_radius: float = Input()
    tip_radius: float = Input()

    @Part
    def box(self):
        # use of `pass_down` is equivalent to `height=self.height, length=…`
        return Box(pass_down='height, length, width',
                   centered=True,  # check in the class Box definition the effect of setting centered to False
                   color="green")

    @Part
    def cone(self):
        return Cone(radius1=self.base_radius, radius2=self.tip_radius, height=self.cone_height,  # angle=math.pi * 3/2,
                    position=translate(self.position, "z", -0.75))


    # ------------------------------- boolean operations -----------------------
    # to perform boolean operations you always need one shape_in solid and one or a list of tools

    @Part
    def fused_solid(self):
        """Union of the two solids"""
        return FusedSolid(shape_in=self.box, tool=self.cone, color="Orange")

    @Part
    def box_less_cone(self):
        """Subtracts the tool solid from the shape_in solid."""
        return SubtractedSolid(shape_in=self.box, tool=self.cone)

    @Part
    def cone_less_box(self):
        """This will fail, because the result is not a single solid.
        See what happens when you move the cone to avoid this"""
        return SubtractedSolid(shape_in=self.cone,
                               tool=self.box,
                               label='cone less box fails!')

    @Part
    def intersection_box_cone(self):
        """Intersection of the two solids (the common part)"""
        return CommonSolid(shape_in=self.box, tool=self.cone, color="red")

    @Part
    def partitioned_solid(self):
        """Splits solids through their intersection. Returns a list of solids"""
        return PartitionedSolid(solid_in=self.box, tool=self.cone,
                                keep_tool=True)  # False by default

    @Part
    def cutting_plane(self):
        """Defining a regular plane to use later"""
        return Plane(reference=self.box.cog, normal=VZ)

    @Part
    def cutting_plane_1(self):
        """Same as above, just different method"""
        return Plane(reference=self.box.cog, normal=Vector(0, 0, 1), color="red",
                     v_dim=max(self.width, self.length) * 1.1)  #v_dim: display plane larger than box, so we can see it

    @Part
    def half_space_solid(self):
        """The plane above is used to build an infinite solid. in this case, everything below cutting_plane.
        observe how it is displayed in the GUI, and what happens when you zoom in to the cube and cone while showing it"""
        return HalfSpaceSolid(built_from=self.cutting_plane,
                              point=self.box.cog.translate('z', -0.2)
                              )

    @Part
    def fused_less_halfspacesolid(self):
        """using the half space solid to obtain the top part of the fused solid.
         Try yourself to obtain also the lower part of fused_solid!"""
        return SubtractedSolid(shape_in=self.fused_solid,
                               tool=self.half_space_solid
                               )


if __name__ == '__main__':
    from parapy.gui import display

    booleans = BuildingSolids(height=1, length=1, width=1.1,
                         cone_height=1.5, base_radius=0.5, tip_radius=0.2,
                         label='testing booleans')
    display(booleans)