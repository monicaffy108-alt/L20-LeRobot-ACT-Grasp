"""Map MediaPipe hand landmarks to the 16 L20 joint values.

Landmarks are normalized to [0, 1], so the mapping is scale invariant. The
fingers are measured in the image plane, which foreshortens joints that rotate
towards the camera; the heuristics below compensate for that.
"""

from __future__ import annotations

import math

from .joints import NUM_JOINTS, validate

# MediaPipe landmark indices.
WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = {"index": INDEX, "middle": MIDDLE, "ring": RING, "pinky": PINKY}

LANDMARK_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# A straight finger measures close to 180 degrees between consecutive
# landmarks; a closed fist measures well under 100 in the 2D projection.
FLEX_STRAIGHT_DEG = 180.0
FLEX_BENT_DEG = 75.0

ABD_GAIN = 2.5
ABD_CENTER = 50.0

THUMB_SPREAD_MIN_DEG = 20.0
THUMB_SPREAD_MAX_DEG = 70.0


def _angle(a, b, c) -> float:
    """Angle at ``b`` formed by the points ``a-b-c``, in degrees."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return FLEX_STRAIGHT_DEG
    dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _signed_spread(origin, reference, point) -> float:
    """Degrees from ``reference`` to ``point`` around ``origin``, signed by side."""
    ref = (reference[0] - origin[0], reference[1] - origin[1])
    vec = (point[0] - origin[0], point[1] - origin[1])
    n_ref = math.hypot(*ref)
    n_vec = math.hypot(*vec)
    if n_ref < 1e-6 or n_vec < 1e-6:
        return 0.0
    dot = (ref[0] * vec[0] + ref[1] * vec[1]) / (n_ref * n_vec)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    cross = ref[0] * vec[1] - ref[1] * vec[0]
    return -angle if cross < 0 else angle


def _flex_to_value(angle_deg: float) -> float:
    """Map a measured joint angle to 0-100, where 100 is fully extended."""
    span = FLEX_STRAIGHT_DEG - FLEX_BENT_DEG
    value = (angle_deg - FLEX_BENT_DEG) / span * 100.0
    return max(0.0, min(100.0, value))


def landmarks_to_joints(points) -> list[float]:
    """Convert 21 normalized landmarks into the 16 L20 joint values."""
    values = [0.0] * NUM_JOINTS

    # Thumb: abd, yaw, root1, tip.
    spread = _signed_spread(points[WRIST], points[INDEX[0]], points[THUMB[0]])
    opening = (abs(spread) - THUMB_SPREAD_MIN_DEG) / (
        THUMB_SPREAD_MAX_DEG - THUMB_SPREAD_MIN_DEG
    ) * 100.0
    values[0] = max(0.0, min(100.0, opening))
    values[1] = _flex_to_value(_angle(points[WRIST], points[THUMB[0]], points[THUMB[1]]))
    values[2] = _flex_to_value(_angle(points[THUMB[0]], points[THUMB[1]], points[THUMB[2]]))
    values[3] = _flex_to_value(_angle(points[THUMB[1]], points[THUMB[2]], points[THUMB[3]]))

    # Four long fingers: abd, root1, tip.
    # root1 is the MCP joint, tip is the PIP joint. The DIP joint is a URDF
    # mimic and is not measured separately.
    # 2D projection foreshortens MCP rotation, so root1 takes the smaller of the
    # MCP and PIP estimates: whichever joint is most bent drives the command.
    middle_mcp = points[MIDDLE[0]]
    for offset, name in enumerate(("index", "middle", "ring", "pinky")):
        mcp, pip, dip, _tip = FINGERS[name]
        base = 4 + offset * 3
        mcp_flex = _flex_to_value(_angle(points[WRIST], points[mcp], points[pip]))
        pip_flex = _flex_to_value(_angle(points[mcp], points[pip], points[dip]))
        values[base + 1] = min(mcp_flex, pip_flex)
        values[base + 2] = pip_flex
        # Lateral spread is only meaningful while the finger is extended: once it
        # curls towards the camera the projected knuckles collapse onto each other
        # and the measured angle runs to an extreme. Fade it out with flexion.
        extension = pip_flex / 100.0
        lateral = _signed_spread(points[WRIST], middle_mcp, points[mcp])
        raw_abd = max(0.0, min(100.0, ABD_CENTER + ABD_GAIN * lateral))
        values[base] = ABD_CENTER + (raw_abd - ABD_CENTER) * extension
    return validate(values)


class Smoother:
    """Exponential moving average with a per-update change limit."""

    def __init__(self, alpha: float = 0.5, max_step: float = 12.0):
        self.alpha = alpha
        self.max_step = max_step
        self._state: list[float] | None = None

    def reset(self, values) -> None:
        self._state = list(values)

    def update(self, values) -> list[float]:
        values = list(values)
        if self._state is None:
            self._state = values
            return list(values)
        out = []
        for previous, new in zip(self._state, values):
            blended = previous + self.alpha * (new - previous)
            delta = max(-self.max_step, min(self.max_step, blended - previous))
            out.append(previous + delta)
        self._state = out
        return list(out)
