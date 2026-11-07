# L20 LeRobot ACT Grasp

<p align="center">
  <img src="https://img.shields.io/badge/status-work_in_progress-yellow" alt="work in progress">
  <img src="https://img.shields.io/badge/framework-LeRobot-yellow" alt="LeRobot">
  <img src="https://img.shields.io/badge/policy-ACT-blue" alt="ACT">
  <img src="https://img.shields.io/badge/hardware-LinkerHand_L20-orange" alt="LinkerHand L20">
  <img src="https://img.shields.io/badge/training-RTX_5070_12GB-green" alt="RTX 5070">
  <img src="https://img.shields.io/badge/sensor-RealSense_D405-teal" alt="RealSense D405">
</p>

Vision-conditioned grasping on the LinkerHand L20, a 16-DoF dexterous hand, trained with ACT
(Action Chunking with Transformers) through [LeRobot](https://github.com/huggingface/lerobot).

## Pipeline

1. **Demonstrate** — an operator hand drives the L20 through the MediaPipe teleoperation layer
   from [l20-gesture-sequencer](https://github.com/monicaffy108-alt/l20-gesture-sequencer)
2. **Record** — RGB frames from an Intel RealSense D405 plus the 16 commanded joint angles,
   stored as a LeRobot dataset
3. **Train** — ACT on a local RTX 5070 (12 GB)
4. **Deploy** — the trained policy closes the loop on the real hand, conditioned on what the
   camera sees

## Second camera (operator view)

The RealSense watches the robot hand. A second camera watches the operator's hand, so MediaPipe
can estimate joint angles. The two views must stay separate: if the policy is trained on frames
that contain a human hand, it learns to react to the operator instead of the object.

When the capture machine has no second camera, a laptop on the same network can serve its webcam
as an MJPEG stream:

```bash
# on the laptop
pip install opencv-python numpy
python scripts/laptop_camera.py --camera 0 --port 8080
# prints: open on desktop : http://10.10.12.175:8080/video

# on the capture machine
import cv2
cap = cv2.VideoCapture("http://10.10.12.175:8080/video")
```

The laptop needs no cable to the capture machine, only the same router. Allow port 8080 through
the laptop firewall the first time it runs.

On macOS, install into a virtual environment so Homebrew's Python does not block `pip`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install opencv-python numpy
python scripts/laptop_camera.py
```

macOS asks for camera access on the first run; grant it to the terminal application. If the
prompt was dismissed, re-enable it under System Settings → Privacy & Security → Camera.

## Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install lerobot mediapipe linkerbot pyrealsense2
```

## Record demonstrations

```bash
python -m l20act.record \
  --repo-id you/l20-grasp \
  --task "pick up the block" \
  --operator-url http://10.10.10.234:8080/video \
  --episodes 20 --seconds 30
```

The operator's hand drives the hand; the D405 records what the policy will see. Press `d` to
finish an episode early, `q` to quit. Between episodes the terminal waits for Enter so the object
can be repositioned.

Add `--dry-run` to exercise the whole loop without sending CAN traffic, and `--no-preview` on a
headless machine.

## Dataset schema

| Key | Shape | Meaning |
|---|---|---|
| `observation.images.robot` | 480 × 640 × 3 | D405 colour frame, the policy input |
| `observation.state` | 16 | measured joint angles (0–100) |
| `action` | 16 | commanded joint angles (0–100) |
| `task` | string | task description, stored per frame |

Two decisions worth stating:

- **Only one camera enters the dataset.** The operator view is deliberately excluded — a policy
  trained on frames containing a human hand learns to depend on it, and no human hand exists at
  deployment.
- **Angles stay in the hardware's native 0–100 units.** LeRobot normalises internally, so
  deploying a predicted action means calling `set_angles()` directly; no unit conversion sits
  between the policy output and the CAN frame, where an error would be hard to see.

## Status

Recording pipeline verified end to end: 30 frames captured at a locked 10.0 fps in dry-run mode,
dataset reloads with the expected shapes. Training and evaluation are next.
