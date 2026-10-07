# vla-seed-variance

How much do evaluation and training seeds move the success rate of vision-language-action (VLA) policies on LIBERO, and what in the simulator moves with them.

## Reference cell: released SmolVLA on LIBERO-Spatial

Checkpoint `HuggingFaceVLA/smolvla_libero`, LeRobot's LIBERO evaluation (one executed action per step, batch of 50 environments per task, one process), the 50 initial states of each of the 10 tasks (500 episodes), run twice with evaluation seeds 1000 and 1001.

| Evaluation seed | Successes | Success rate (95% Wilson) |
|---|---|---|
| 1000 | 354 / 500 | 70.8 [66.7, 74.6] |
| 1001 | 348 / 500 | 69.6 [65.4, 73.5] |

146 of 500 paired episodes (29.2%) change outcome between the two seeds. Initial-state hashes match in all 500 pairs, but first-frame hashes differ in all 500.

## Why the first frames differ

At every reset LIBERO re-samples the placement of fixed furniture (wooden cabinet, flat stove) and writes it into the MuJoCo model, not into the simulator state. Setting the stored initial state restores the movable objects and the robot but not the furniture, so the evaluation seed moves the furniture by a few millimetres. `scripts/fixture_pin_test.py` shows that with the furniture pinned to the same placement, the frames for two different seeds are pixel-identical: furniture placement is the only seed-dependent part of the scene.

## Files

- `scripts/run_lerobot_libero_hashed.py`: runs `lerobot-eval` unchanged and logs every reset with the initial-state hash and a hash of each camera's first frame.
- `scripts/compare_reruns.py`: pairs two runs by task and initial state; success rates, flips, hash checks.
- `scripts/run_reference_cell.sh`: the two runs, the comparison and a push notification (`NTFY_TOPIC`).
- `scripts/fixture_pin_test.py`: first-frame comparison with and without pinned furniture.
- `results/`: per-run evaluation output, reset logs, paired episodes and test reports.
