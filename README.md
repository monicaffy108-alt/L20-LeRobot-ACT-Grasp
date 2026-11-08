# L20 LeRobot ACT Grasp

<p align="center">
  <img src="https://img.shields.io/badge/status-grasping_on_hardware-brightgreen" alt="grasping on hardware">
  <img src="https://img.shields.io/badge/framework-LeRobot-yellow" alt="LeRobot">
  <img src="https://img.shields.io/badge/policy-ACT-blue" alt="ACT">
  <img src="https://img.shields.io/badge/hardware-LinkerHand_L20-orange" alt="LinkerHand L20">
  <img src="https://img.shields.io/badge/training-RTX_5070_12GB-green" alt="RTX 5070">
  <img src="https://img.shields.io/badge/sensor-RealSense_D405-teal" alt="RealSense D405">
</p>

Vision-conditioned grasping on the LinkerHand L20, a 16-DoF dexterous hand, trained with ACT
(Action Chunking with Transformers) through [LeRobot](https://github.com/huggingface/lerobot).

**Demo: [the trained policy grasps a tissue pack, 5/5 runs](https://youtube.com/shorts/iZnRkgDtTtE)**

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

Recording runs at 8 fps by default. MediaPipe inference (~50 ms) and the CAN write (~32 ms) put
the ceiling at about 9 fps on this machine, so asking for a higher rate only makes the timestamps
disagree with the frames; the script says so when that happens.

Add `--dry-run` to exercise the whole loop without sending CAN traffic, and `--no-preview` on a
headless machine. Add `--resume` to append to an existing dataset; without it `create()` starts
over, because it does not check whether the directory is already there.

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

## Train

```bash
# a small wrapper: LeRobot marks the newest checkpoint with a symlink, which needs
# admin rights on Windows; this falls back to a directory junction
python scripts/train_act.py --dataset.repo_id=local/l20-grasp --dataset.root=datasets/l20-grasp \
  --policy.type=act --policy.repo_id=local/act-l20-grasp --policy.push_to_hub=false \
  --output_dir=outputs/act_l20 --steps=20000 --batch_size=8 --num_workers=4 \
  --save_freq=5000 --trackio.enable=false
```

## Deploy

```bash
python -m l20act.eval --checkpoint outputs/act_l20/checkpoints/last/pretrained_model --seconds 30
```

The policy observes only what it saw in training — the D405 frame and the measured joint
angles; MediaPipe and the operator camera are not involved. ACT predicts a 30-action chunk,
but the queue is dropped every 25 steps so the policy re-observes the scene instead of
replaying open-loop. A per-frame rate limit identical to the recording smoother caps each
command. Press `q` in the terminal to stop, add `--dry-run` to watch the commands without
CAN traffic.

## Status

The loop is closed: the policy grasps a tissue pack on the real hand, **5/5 consecutive
successful grasps** in closed-loop evaluation (~2.2 control steps/s, object re-placed between
runs) — [video](https://youtube.com/shorts/iZnRkgDtTtE). Final model: 25 clean single-grasp demonstrations (2000 frames at 8 fps), ACT with
chunk size 30 trained for 20000 steps (L1 loss 0.43 → 0.09) on a single RTX 5070.

Getting there took three dataset iterations, which is the real lesson of this repo:

- **v1/v2 — camera pointed at the operator**, not the robot. The policy learned to react to
  a human face. Fixing the camera aim mattered more than any hyperparameter.
- **v3 — correct view, messy demos** (repeated open-close flapping in every episode). The
  policy faithfully learned the rhythm but never made contact.
- **v4 — one clean grasp per episode** (open → object to the palm → close → hold), plus a
  shorter action chunk (100 → 30). The rhythm became an actual grip.
