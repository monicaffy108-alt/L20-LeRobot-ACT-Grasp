"""Joint order and validation for the LinkerHand L20.

The 16 active joints follow the order the SDK uses across CAN frames 0x41-0x45.
Every value is in the 0-100 range the hardware accepts, where 100 is fully
extended and 0 is fully flexed.
"""

from __future__ import annotations

JOINT_NAMES = (
    "thumb_abd", "thumb_yaw", "thumb_root1", "thumb_tip",
    "index_abd", "index_root1", "index_tip",
    "middle_abd", "middle_root1", "middle_tip",
    "ring_abd", "ring_root1", "ring_tip",
    "pinky_abd", "pinky_root1", "pinky_tip",
)

NUM_JOINTS = len(JOINT_NAMES)


def validate(values) -> list[float]:
    """Coerce to a list of 16 floats inside [0, 100]."""
    values = [float(value) for value in values]
    if len(values) != NUM_JOINTS:
        raise ValueError(f"expected {NUM_JOINTS} joint values, got {len(values)}")
    return [max(0.0, min(100.0, value)) for value in values]
