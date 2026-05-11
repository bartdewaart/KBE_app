from parapy.core import Input, Part, Sequence, Attribute
from parapy.geom import LoftedSurface
from .blade_section import BladeSection
from .config import get_value

class Blade(LoftedSurface):
    n_segments = Input()
    is_ruled = Input(True)
    is_approx = Input(True)
    is_solid = Input(False)

    @Attribute
    def profiles(self):
        """Required by LoftedSurface to build the skin."""
        curves = [s.positioned_curve for s in self.sections]
        if get_value("debug", "loft", default=False):
            print(f"DEBUG: Lofting with {len(curves)} profiles")
            for i, section in enumerate(self.sections):
                curve = curves[i]
                try:
                    n_ctrl = len(getattr(curve, "control_points", []))
                except Exception:
                    n_ctrl = "?"
                print(
                    f"  Section {i}: r={section.radius:.4f}, chord={section.chord:.4f}, "
                    f"pitch={section.pitch_deg:.2f}, n_ctrl={n_ctrl}"
                )
        return curves

    @Part
    def sections(self):
        """Sequence MUST be a Part for visualization."""
        return Sequence(type=BladeSection, quantify=self.n_segments)