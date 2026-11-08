"""Run a trained ACT policy on the real L20 hand.

The loop mirrors the recording setup exactly: the D405 frame and the measured
joint angles go in, 16 commanded angles come out. No MediaPipe, no operator
camera — the policy has never seen a human hand and must not see one now.

    python -m l20act.eval --seconds 30
    python -m l20act.eval --dry-run --seconds 5   # no CAN traffic

ACT predicts a chunk of 100 actions at once. Replaying all 100 open-loop is
fragile, so the queue is dropped every ``--replan-steps`` and the policy
re-observes the scene. ``--max-step`` caps how far one command may move a
joint, matching the rate limit used while recording.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .cameras import RobotCamera
from .hand import open_hand
from .joints import validate
from .record import IMAGE_KEY, pressed

DEFAULT_CHECKPOINT = "outputs/act_l20/checkpoints/last/pretrained_model"


def load_policy(checkpoint: str):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(checkpoint)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=checkpoint
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device).eval()
    return policy, preprocessor, postprocessor, device


def to_batch(frame: np.ndarray, state: list[float], device: str) -> dict:
    image = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    joints = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    return {
        IMAGE_KEY: image.to(device),
        "observation.state": joints.to(device),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="l20act.eval",
        description="Deploy a trained ACT policy on the L20 hand",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="how long the policy controls the hand; press q to stop earlier")
    parser.add_argument("--fps", type=int, default=8, help="control rate; match the training fps")
    parser.add_argument("--replan-steps", type=int, default=25,
                        help="drop the action chunk and re-observe every N steps")
    parser.add_argument("--max-step", type=float, default=12.0,
                        help="max joint change per frame in 0-100 units")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--interface-type", default="pcan")
    parser.add_argument("--interface-name", default="PCAN_USBBUS1")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="no hardware; print actions only")
    args = parser.parse_args(argv)

    print(f"loading policy: {args.checkpoint}")
    policy, preprocessor, postprocessor, device = load_policy(args.checkpoint)
    print(f"policy ready on {device} (chunk {policy.config.chunk_size}, "
          f"replanning every {args.replan_steps} steps)")

    robot_cam = RobotCamera(width=args.width, height=args.height)
    hand = open_hand(
        dry_run=args.dry_run,
        side=args.side,
        interface_type=args.interface_type,
        interface_name=args.interface_name,
    )

    print("\nplace the object, keep your fingers clear, then press Enter (Ctrl+C to abort)...",
          end="", flush=True)
    try:
        input()
    except KeyboardInterrupt:
        print("\naborted")
        return 1

    steps = 0
    try:
        with hand:
            previous = hand.read_angles()
            started = time.monotonic()
            while True:
                now = time.monotonic()
                if now - started >= args.seconds or pressed("qQ"):
                    break

                frame = robot_cam.read()
                if frame is None:
                    continue
                state = hand.read_angles()

                if steps % args.replan_steps == 0:
                    policy.reset()

                batch = to_batch(frame, state, device)
                action = postprocessor(policy.select_action(preprocessor(batch)))
                target = validate(action.squeeze(0).detach().cpu().tolist())

                # rate-limit the jump from the last command, same cap as recording
                target = [
                    float(np.clip(t, p - args.max_step, p + args.max_step))
                    for t, p in zip(target, previous)
                ]
                hand.set_angles(target)
                previous = target
                steps += 1

                if not args.no_preview:
                    import cv2

                    view = frame.copy()
                    cv2.putText(view, f"POLICY step {steps}", (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("L20 eval - q in the terminal quits", view)
                    cv2.waitKey(1)

                delay = started + steps / args.fps - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        robot_cam.close()
        if not args.no_preview:
            import cv2

            cv2.destroyAllWindows()

    elapsed = time.monotonic() - started
    print(f"\n{steps} control steps in {elapsed:.1f}s ({steps / max(elapsed, 1e-6):.1f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
