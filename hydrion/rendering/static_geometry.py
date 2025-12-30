"""
Static spatial context for truthful visualization.

Geometry in this file:
- is visual-only
- never affects mechanics
- never mutates particles
- exists solely for human interpretability
"""


# ----------------------------
# Geometry constants
# ----------------------------

INLET_Y = 1.0
INLET_X_RANGE = (-0.4, 0.4)

CAPTURE_REGION = {
    "x": 0.8,
    "y_top": 0.6,
    "y_bottom": -0.2,
}

CHAMBER = {
    "x": 0.8,
    "y": -0.5,
    "width": 0.3,
    "height": 0.3,
}


# ----------------------------
# Drawing helpers
# ----------------------------

def draw_static_context(renderer):
    """
    Draw all static geometry.
    This function must be side-effect free.
    """

    renderer.draw_inlet(
        x_range=INLET_X_RANGE,
        y=INLET_Y,
    )

    renderer.draw_capture_region(
        x=CAPTURE_REGION["x"],
        y_top=CAPTURE_REGION["y_top"],
        y_bottom=CAPTURE_REGION["y_bottom"],
    )

    renderer.draw_chamber(
        x=CHAMBER["x"],
        y=CHAMBER["y"],
        width=CHAMBER["width"],
        height=CHAMBER["height"],
    )
