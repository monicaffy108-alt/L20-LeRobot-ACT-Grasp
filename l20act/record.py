"""Record grasping demonstrations into a LeRobot dataset.

The operator's hand drives the L20 through a camera; the D405 observes the hand
and the object. Every frame stores:

    observation.images.robot   the D405 image (what the policy sees)
    observation.state          measured joint angles (0-100)
    action                     commanded joint angles (0-100)

The operator camera never enters the dataset. If the policy saw a human hand it
would learn to depend on it, and no human hand exists at deployment time.

Run it:
    python -m l20act.record --repo-id you/l20-grasp --task "pick up the block"
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .cameras import DEFAULT_MODEL, HandTracker, RobotCamera, ThreadedCamera
from .hand import open_hand
from .joints import JOINT_NAMES, NUM_JOINTS
from .mapping import LANDMARK_CONNECTIONS, Smoother, landmarks_to_joints

IMAGE_KEY = "observation.images.robot"

try:  # Windows console keyboard polling; absent on Linux/macOS
    import msvcrt as _MSVCRT
except ImportError:
    _MSVCRT = None


def build_features(width: int, height: int) -> dict:
    return {
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (NUM_JOINTS,), "names": list(JOINT_NAMES)},
        "action": {"dtype": "float32", "shape": (NUM_JOINTS,), "names": list(JOINT_NAMES)},
    }


def wait_for_enter(prompt: str) -> None:
    print(prompt, end="", flush=True)
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        raise SystemExit("\naborted")


def pressed(chars: str) -> bool:
    """True when one of ``chars`` was typed in the terminal."""
    if _MSVCRT is None:
        return False
    while _MSVCRT.kbhit():
        if _MSVCRT.getwch() in chars:
            return True
    return False


def draw_preview(robot_frame, operator_frame, points, state, recording: bool):
    import cv2

    robot = robot_frame.copy()
    label = "RECORDING" if recording else "idle"
    colour = (0, 0, 255) if recording else (200, 200, 200)
    cv2.putText(robot, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    cv2.putText(robot, f"grip {100 - min(state):5.1f}", (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if operator_frame is not None:
        operator = operator_frame.copy()
        height, width = operator.shape[:2]
        if points:
            pixels = [(int(x * width), int(y * height)) for x, y in points]
            for a, b in LANDMARK_CONNECTIONS:
                cv2.line(operator, pixels[a], pixels[b], (0, 200, 0), 2, cv2.LINE_AA)
            for point in pixels:
                cv2.circle(operator, point, 3, (0, 0, 255), -1, cv2.LINE_AA)
        operator = cv2.resize(operator, (robot.shape[1], robot.shape[0]))
        canvas = np.hstack([robot, operator])
    else:
        canvas = robot
    cv2.imshow("L20 record - d finishes the episode, q quits", canvas)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="l20act.record",
        description="Teleoperate the L20 with your hand and record LeRobot episodes",
    )
    parser.add_argument("--repo-id", default="local/l20-grasp")
    parser.add_argument("--root", default=None, help="dataset directory (default: ./datasets/<repo-id>)")
    parser.add_argument("--task", default="pick up the object",
                        help="task string stored with every frame")
    parser.add_argument("--episodes", type=int, default=1, help="number of episodes to record")
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="max length of one episode; press d to finish earlier")
    parser.add_argument("--fps", type=int, default=15, help="recording rate")
    parser.add_argument("--operator-url", required=True,
                        help="MJPEG URL of the camera watching your hand, e.g. http://10.0.0.5:8080/video")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--alpha", type=float, default=0.5, help="smoothing 0-1; lower is smoother")
    parser.add_argument("--max-step", type=float, default=12.0,
                        help="max joint change per frame in 0-100 units")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="MediaPipe hand landmarker")
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--interface-type", default="pcan")
    parser.add_argument("--interface-name", default="PCAN_USBBUS1")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="no hardware; echo commands only")
    args = parser.parse_args(argv)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = args.root or f"datasets/{args.repo_id.replace('/', '_')}"
    features = build_features(args.width, args.height)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=features, root=root, use_videos=True
    )

    smoother = Smoother(alpha=args.alpha, max_step=args.max_step)
    target = [50.0] * NUM_JOINTS
    smoother.reset(target)

    robot_cam = RobotCamera(width=args.width, height=args.height)
    operator_cam = ThreadedCamera(args.operator_url, width=args.width, height=args.height)
    tracker = HandTracker(args.model)
    hand = open_hand(
        dry_run=args.dry_run,
        side=args.side,
        interface_type=args.interface_type,
        interface_name=args.interface_name,
    )

    recorded = 0
    quit_requested = False
    try:
        with hand:
            for episode in range(1, args.episodes + 1):
                if quit_requested:
                    break
                wait_for_enter(
                    f"\nEpisode {episode}/{args.episodes}: place the object, then press Enter "
                    "(Ctrl+C to stop)... "
                )
                for countdown in (3, 2, 1):
                    print(f"  {countdown}...", end="", flush=True)
                    time.sleep(1.0)
                print(" recording")

                started = time.monotonic()
                next_frame = started
                frames = 0
                lost = False

                while True:
                    now = time.monotonic()
                    if now - started >= args.seconds:
                        break

                    robot_frame = robot_cam.read()
                    if robot_frame is None:
                        continue
                    if pressed("qQ"):
                        quit_requested = True
                        break
                    if pressed("dD"):
                        break

                    operator_frame = operator_cam.read()
                    points = None
                    if operator_frame is not None:
                        import cv2

                        rgb = cv2.cvtColor(operator_frame, cv2.COLOR_BGR2RGB)
                        points = tracker.detect(rgb, int(now * 1000) % 2_000_000_000)
                    lost = points is None
                    if points is not None:
                        target = smoother.update(landmarks_to_joints(points))

                    hand.set_angles(target)
                    state = hand.read_angles()

                    dataset.add_frame({
                        IMAGE_KEY: robot_frame,
                        "observation.state": np.asarray(state, dtype=np.float32),
                        "action": np.asarray(target, dtype=np.float32),
                        "task": args.task,
                    })
                    frames += 1

                    if not args.no_preview:
                        draw_preview(robot_frame, operator_frame, points, state, True)

                    next_frame += 1.0 / args.fps
                    delay = next_frame - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)

                elapsed = time.monotonic() - started
                if lost:
                    print("  warning: the operator hand was lost during recording")
                print(f"  {frames} frames in {elapsed:.1f}s ({frames / max(elapsed, 1e-6):.1f} fps)")

                if frames < args.fps:
                    print("  too short to be useful, discarding")
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded += 1
                print(f"  saved: {recorded} episode(s), {dataset.num_frames} frames total")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        tracker.close()
        operator_cam.close()
        robot_cam.close()
        if not args.no_preview:
            import cv2

            cv2.destroyAllWindows()

    print(f"\ndataset: {root}")
    print(f"episodes: {dataset.num_episodes}   frames: {dataset.num_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
