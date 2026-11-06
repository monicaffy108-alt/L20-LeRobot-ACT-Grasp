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

## Status

Work in progress. The dataset schema, training config and evaluation protocol are being built.
