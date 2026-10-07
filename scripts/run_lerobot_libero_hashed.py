"""Run LeRobot's LIBERO evaluation unchanged and log every environment reset.

Same as run_lerobot_libero_logged.py in libero-vla-eval, plus a SHA-1 hash of the first rendered
frame of each camera returned by reset, so reruns can be checked for identical scenes.

Usage: RESET_LOG=/path/resets.jsonl python run_lerobot_libero_hashed.py <lerobot-eval arguments>
"""
import hashlib
import inspect
import json
import os

import numpy as np

import lerobot.envs.libero as libero_env

LOG = open(os.environ["RESET_LOG"], "a")
_original_reset = libero_env.LiberoEnv.reset


def sha1_array(x):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def state_hash(state):
    return sha1_array(np.asarray(state, dtype=np.float64))


def frame_hashes(obs):
    pixels = obs.get("pixels") if isinstance(obs, dict) else None
    if isinstance(pixels, dict):
        return {k: sha1_array(v) for k, v in sorted(pixels.items())}
    if pixels is not None:
        return {"pixels": sha1_array(pixels)}
    return None


def reset(self, seed=None, **kwargs):
    states = self._init_states if self.init_states else None
    idx = self.init_state_id % len(states) if states is not None else None
    caller = "autoreset" if any(f.function == "step" for f in inspect.stack()[1:8]) else "explicit"
    out = _original_reset(self, seed=seed, **kwargs)
    obs = out[0] if isinstance(out, tuple) else out
    LOG.write(json.dumps({
        "task_id": self.task_id,
        "sub_env": self.episode_index,
        "init_state_idx": idx,
        "init_state_sha1": state_hash(states[idx]) if states is not None else None,
        "first_frame_sha1": frame_hashes(obs),
        "caller": caller,
        "seed": seed,
    }) + "\n")
    LOG.flush()
    return out


libero_env.LiberoEnv.reset = reset

from lerobot.scripts.lerobot_eval import main  # noqa: E402

if __name__ == "__main__":
    main()
