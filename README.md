# Evaluation seed on LIBERO: policy noise versus furniture placement

How much does the evaluation seed move the result of a vision-language-action (VLA) policy on LIBERO, and how much of that comes from the policy's own sampling noise versus the placement of fixed furniture in the scene?

In `lerobot-eval`, one `--seed` drives both: it seeds the policy's random noise and it also moves the cabinet and stove that LIBERO places at every reset. This repository separates the two for the released SmolVLA checkpoint.

## Setup

- Checkpoint [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero) (revision 6721902), LeRobot's LIBERO evaluation (`lerobot-eval`, batch of 50 environments per task, one executed action per prediction, AMP off), MuJoCo 3.3.7, unmodified LIBERO textures.
- LIBERO-Spatial, LIBERO-Object and LIBERO-Goal: the 50 stored initial states of each of the 10 tasks, 500 episodes per suite.
- Cells:
  - A: seed 1000.
  - B: seed 1001 (policy noise and furniture placement both change). On Spatial, B was run a second time with a runner that also logs the policy's noise stream; that stream matches cell C on all 10 tasks, so C vs B changes only the furniture. The Spatial numbers below use this rerun.
  - C: policy seed 1001 on the furniture placement of seed 1000 (only the policy noise changes). Not run on Object, which has no fixed furniture.
  - Same-seed reruns, to measure the compute floor.
  - Deterministic control: OpenVLA-7B on LIBERO-Spatial, environment seed 0 against 1 (200 episodes).

## Results

Flips are paired episodes (same task and initial state) whose outcome changes between two cells.

| Suite | Success A / B | Flips A vs B (both) | Flips A vs C (noise only) | Flips C vs B (furniture only) | Same-seed rerun |
|---|---|---|---|---|---|
| Spatial | 354 / 349 | 149 / 500 (29.8%) | 138 / 500 (27.6%) | 83 / 500 (16.6%) | 15 / 500 |
| Goal | 375 / 368 | 115 / 500 (23.0%) | 103 / 500 (20.6%) | 72 / 500 (14.4%) | 17 / 500 |
| Object | 390 / 385 | 105 / 500 (21.0%) | not applicable | not applicable | 0 / 500 |

The first Spatial B run (348 / 500) differs from the rerun in 9 of 500 episodes, below the same-seed floor; with it, A vs B reads 146 and C vs B 86.

Furniture term = flips(A vs B) minus flips(A vs C), paired bootstrap over episodes (20,000 draws):

| Suite | Furniture term (points) | 95% CI | One-sided 95% upper bound |
|---|---|---|---|
| Spatial | +2.2 | [-1.4, +5.8] | 5.2 |
| Goal | +2.4 | [-0.8, +5.8] | 5.2 |
| Spatial and Goal pooled | +2.3 | [-0.2, +4.7] | 4.3 |

With the first Spatial B run: Spatial +1.6 [-2.0, +5.2], upper bound 4.6; pooled +2.0 [-0.5, +4.4], upper bound 4.1.

Reading: changing the seed flips about a fifth to a third of the episodes. The two things the seed controls do not add up. On Spatial, changing only the policy noise flips 27.6% of the episodes, moving only the furniture flips 16.6%, and changing both flips 29.8%, because both changes flip mostly the same fragile episodes: 36 episodes flip under the noise-only change and under the furniture-only change alike, against 22.9 if the two were independent (Goal: 30 against 14.8). The furniture term is therefore what moving the furniture adds once the noise has already changed, at most about 4-5 points; it is not the share of the seed effect that comes from the furniture. The scene itself is sensitive: moving the furniture alone, with the noise held fixed, flips 14-17% of the episodes. Exploratory, Spatial only: moving the furniture flips 22.7% of the episodes in the three tasks whose bowl sits in or on furniture (cabinet drawer, stove, cabinet top) against 14.0% in the other seven, while changing the noise flips 28.7% against 27.1%. On the deterministic OpenVLA-7B, the environment seed alone flips 40 / 200 episodes (20.0%; success 80.5% to 86.5%, McNemar p = 0.081).

Scope: one stochastic checkpoint and one deterministic control, in one pinned pipeline. The success levels sit below published SmolVLA numbers; every quantity here is a difference inside the same pipeline.

## Why the first frames differ between seeds

At every reset LIBERO re-samples the placement of fixed furniture (wooden cabinet, flat stove) and writes it into the MuJoCo model, not into the simulator state. Setting the stored initial state restores the movable objects and the robot but not the furniture, so the evaluation seed moves the furniture (between seeds 1000 and 1001 on Spatial, the largest furniture shift is 12 mm in the median episode and 23 mm at most). `scripts/fixture_pin_test.py` shows that with the furniture pinned, the first frames for two different seeds are pixel-identical: furniture placement is the only seed-dependent part of the scene.

## Files

- `scripts/run_lerobot_libero_hashed.py`: runs `lerobot-eval` unchanged and logs every reset with the initial-state hash and a hash of each camera's first frame.
- `scripts/compare_reruns.py`: pairs two runs by task and initial state; success rates, flips, hash checks.
- `scripts/run_reference_cell.sh`, `scripts/fixture_pin_test.py`: the first Spatial reference pair and the furniture-pinning check.
- `eval_seed_study/scripts/`: the runner chain for all cells, `cell_stats.py` (paired statistics per pair of cells) and `final_analysis.py` (furniture term with bootstrap intervals, and the overlap of noise-only and furniture-only flips).
- `eval_seed_study/results/`: per-cell evaluation output and reset logs; `final_analysis.txt` holds the numbers above. `smolvla_spatial_seed1001_free_logged/` is the Spatial rerun of B (run script `eval_seed_study/scripts/run_spatial_seed1001_free_logged.sh`, runner `run_lerobot_libero_logged.py` in [libero-vla-eval](https://github.com/lhtrongai/libero-vla-eval)); its `compare_vs_pinned.txt` checks the noise stream and furniture against cell C.
- `results/`: the first Spatial reference pair and the furniture-pinning test. `results/reference_spatial/compare.txt` ends with the verdict of the pipeline check set before the study, STOP: the seed-1000 run lands 8 points below the 78.8% published for this checkpoint on Spatial with one executed action (SMILE, arXiv 2608.29432) (70.8%), and the first frames differ in all 500 pairs, which is what moving the furniture does. That is why this study reports only differences inside one pipeline, not success levels.
- Result file names: `a1_500` means one executed action per prediction and 500 episodes; `free` means the furniture follows the evaluation seed, `pinned` means it is held at the placement of seed 1000. Paths inside the logged output (for example the video paths in `eval_info*.json`) are those of the machine the runs used.

## Reproduce the analysis (CPU, standard library only, about 20 seconds)

```bash
python eval_seed_study/scripts/final_analysis.py   # prints eval_seed_study/results/final_analysis.txt
```

`final_analysis.py` reads only the committed files in `eval_seed_study/results/`. `cell_stats.py` and `compare_reruns.py` were run on the GPU machine right after each cell and expect that machine's layout (each run directory next to its `<run>_resets.jsonl` and `<run>.log`).

## Rerun the evaluation (GPU)

The runs use LeRobot's `lerobot-eval` through a logging wrapper that leaves the evaluation unchanged: `scripts/run_lerobot_libero_hashed.py` here for the first Spatial pair, and `scripts/run_lerobot_libero_logged.py` of [libero-vla-eval](https://github.com/lhtrongai/libero-vla-eval), which also logs the policy noise, for the other cells. The libero-vla-eval README lists the LeRobot environment (LeRobot commit `8c920c4`, hf-libero, MuJoCo 3.3.7) and `env/requirements-lerobot-venv.txt`. Every cell uses the same arguments and changes only `--seed` and `SCENE_SEED_SHIFT`:

```bash
# cell B on Spatial: seed 1001, furniture follows the seed
RESET_LOG=runs/spatial_seed1001_free_resets.jsonl SCENE_SEED_SHIFT=0 \
python run_lerobot_libero_logged.py \
  --policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero --env.task=libero_spatial \
  --eval.batch_size=50 --eval.n_episodes=50 --eval.use_async_envs=false \
  --policy.n_action_steps=1 --policy.use_amp=false --policy.device=cuda \
  --seed=1001 --output_dir=runs/spatial_seed1001_free
```

`--seed=1000 SCENE_SEED_SHIFT=0` gives cell A; `--seed=1001 SCENE_SEED_SHIFT=-1` gives cell C (seed-1001 policy noise on the furniture placement of seed 1000). `--env.task` selects the suite. `eval_seed_study/scripts/` holds the batch and watchdog scripts as they ran on one cloud GPU instance; they assume that machine's folder layout (`/workspace/...`) and need `NTFY_TOPIC` set for notifications.

## License

The code in this repository is under the MIT license (see `LICENSE`). SmolVLA, LeRobot and LIBERO are not included and keep their own licenses.
