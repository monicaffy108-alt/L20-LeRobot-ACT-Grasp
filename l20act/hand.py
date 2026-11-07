"""Thin wrapper around the linkerbot L25 hand."""

from __future__ import annotations

from .joints import validate


class LinkerHand:
    """Real hardware backend.

    The SDK polls angle at 60 Hz once polling is started, so read_angles() reads
    a cached snapshot instead of issuing a blocking request that would contend
    with the poll loop on the CAN bus.
    """

    def __init__(self, side: str = "left", interface_type: str = "pcan",
                 interface_name: str = "PCAN_USBBUS1"):
        self.side = side
        self.interface_type = interface_type
        self.interface_name = interface_name
        self._hand = None

    def __enter__(self) -> "LinkerHand":
        from linkerbot import L25
        from linkerbot.hand.l25.events import SensorSource

        try:
            self._hand = L25(
                side=self.side,
                interface_name=self.interface_name,
                interface_type=self.interface_type,
            )
            self._hand.__enter__()
            self._hand.start_polling({SensorSource.ANGLE: 1 / 60})
        except Exception as exc:
            raise SystemExit(
                f"connection failed: {exc}\n"
                "Checklist:\n"
                "  1. Close every program using the CAN adapter "
                "(vendor GUI, PCAN-View, browser control panel)\n"
                f"  2. Confirm the interface name; currently "
                f"{self.interface_type}:{self.interface_name}\n"
                '  3. Query it with: python -c "import can;print(can.detect_available_configs())"'
            ) from None
        return self

    def __exit__(self, *exc) -> None:
        if self._hand is not None:
            self._hand.__exit__(*exc)

    def set_angles(self, angles) -> None:
        self._hand.angle.set_angles(validate(angles))

    def read_angles(self) -> list[float]:
        data = self._hand.angle.get_snapshot()
        if data is None:
            data = self._hand.angle.get_blocking(timeout_ms=500)
        return validate(data.angles.to_list())


class DryHand:
    """No hardware: echoes the last command, so the loop can be tested offline."""

    def __init__(self, *args, **kwargs):
        self._last = [50.0] * 16

    def __enter__(self) -> "DryHand":
        print("[dry-run] no CAN traffic, commands are only echoed")
        return self

    def __exit__(self, *exc) -> None:
        pass

    def set_angles(self, angles) -> None:
        self._last = validate(angles)

    def read_angles(self) -> list[float]:
        return list(self._last)


def open_hand(dry_run: bool = False, **kwargs):
    return DryHand() if dry_run else LinkerHand(**kwargs)
