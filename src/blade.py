from parapy.core import Base, Input, Part, Sequence
from parapy.geom import LoftedSurface
from .blade_section import BladeSection


class Blade(Base):
    n_segments = Input()

    @Part
    def sections(self):
        return Sequence(type=BladeSection, quantify=self.n_segments)

    @Part
    def surface(self):
        """Creates the 3D skin over all section curves."""
        return LoftedSurface(profiles=[s.section_curve for s in self.sections], color="LightBlue")
