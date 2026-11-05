# L20 LeRobot ACT Grasp

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

## Status

Work in progress. The dataset schema, training config and evaluation protocol are being built.
