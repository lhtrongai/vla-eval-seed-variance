"""Does the evaluation seed change the LIBERO scene beyond fixture placement?

Builds LeRobot's LiberoEnv (same settings as the reference cell: 360x360, two cameras,
10 settle steps) for one task and initial state, and compares first frames for:
  seed_a, seed_a again, seed_b, and seed_b with fixtures pinned to seed_a's placement.
If the pinned frame equals seed_a's frame, fixture placement is the only seed-dependent part.
"""
import hashlib
import json
import sys

import numpy as np
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action

TASK, STATE, SEED_A, SEED_B = int(sys.argv[1]), int(sys.argv[2]), 1000, 1001
OUT = "/workspace/runs/fixture_pin_test"
suite = benchmark.get_benchmark_dict()["libero_spatial"]()


def make():
    env = LiberoEnv(task_suite=suite, task_id=TASK, task_suite_name="libero_spatial",
                    obs_type="pixels_agent_pos", observation_width=360, observation_height=360,
                    n_envs=1, episode_index=STATE)
    env._ensure_env()
    return env


def fixture_poses(raw):
    out = {}
    for name, obj in raw.fixtures_dict.items():
        bid = raw.sim.model.body_name2id(obj.root_body)
        out[name] = (raw.sim.model.body_pos[bid].copy(), raw.sim.model.body_quat[bid].copy())
    return out


def first_frame(seed, pin=None):
    """Same steps as LiberoEnv.reset, with an optional fixture override before the init state is set."""
    env = make()
    w = env._env
    w.seed(seed)
    w.reset()
    raw = w.env
    poses = fixture_poses(raw)
    if pin is not None:
        for name, (pos, quat) in pin.items():
            bid = raw.sim.model.body_name2id(raw.fixtures_dict[name].root_body)
            raw.sim.model.body_pos[bid] = pos
            raw.sim.model.body_quat[bid] = quat
    obs = w.set_init_state(env._init_states[STATE])
    for _ in range(env.num_steps_wait):
        obs, _, _, _ = w.step(get_libero_dummy_action())
    pix = env._format_raw_obs(obs)["pixels"]
    frames = {k: np.asarray(v) for k, v in pix.items()}
    env.close()
    return frames, poses


def sha(x):
    return hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:12]


cases = {}
cases["seed_a"] = first_frame(SEED_A)
cases["seed_a_again"] = first_frame(SEED_A)
cases["seed_b"] = first_frame(SEED_B)
cases["seed_b_pinned_to_a"] = first_frame(SEED_B, pin=cases["seed_a"][1])

ref = cases["seed_a"][0]
report = {"task": TASK, "state": STATE}
for name, (frames, poses) in cases.items():
    report[name] = {
        "frame_sha1": {k: sha(v) for k, v in frames.items()},
        "max_abs_pixel_diff_vs_seed_a": {k: int(np.abs(v.astype(int) - ref[k].astype(int)).max()) for k, v in frames.items()},
        "changed_pixels_pct_vs_seed_a": {k: round(100 * float((v != ref[k]).any(-1).mean()), 2) for k, v in frames.items()},
        "fixture_pos_mm": {n: [round(1000 * float(c), 1) for c in p[0]] for n, p in poses.items()},
    }
np.savez_compressed(f"{OUT}/frames_task{TASK}_state{STATE}.npz",
                    **{f"{c}__{k}": v for c, (fr, _) in cases.items() for k, v in fr.items()})
json.dump(report, open(f"{OUT}/report_task{TASK}_state{STATE}.json", "w"), indent=1)
print(json.dumps(report, indent=1))
