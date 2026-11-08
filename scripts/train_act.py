"""Run LeRobot training with a Windows-friendly checkpoint pointer.

LeRobot marks the newest checkpoint by symlinking `checkpoints/last` to it. Creating a
symlink on Windows needs administrator rights or Developer Mode, so a normal run crashes
right after the first checkpoint is written, even though the checkpoint itself is fine.

Directory junctions need no privilege, so this wrapper replaces that single function and
then hands control to LeRobot. Everything else behaves exactly like `lerobot-train`.

    python scripts/train_act.py --dataset.repo_id=local/l20-grasp ...
    python scripts/train_act.py --resume=true --checkpoint_path=outputs/act_l20/checkpoints/005000
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from lerobot.common import train_utils
from lerobot.scripts import lerobot_train


def update_last_checkpoint(checkpoint_dir: Path) -> None:
    """Point `checkpoints/last` at `checkpoint_dir`, falling back to a junction."""
    last = checkpoint_dir.parent / train_utils.LAST_CHECKPOINT_LINK
    if last.is_symlink():
        last.unlink()
    elif last.exists():
        last.rmdir()

    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    try:
        last.symlink_to(relative_target, target_is_directory=True)
    except OSError:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(last), str(checkpoint_dir)],
            check=True,
            capture_output=True,
        )


lerobot_train.update_last_checkpoint = update_last_checkpoint

if __name__ == "__main__":
    sys.exit(lerobot_train.main())
