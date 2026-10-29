#!/usr/bin/env bash
# SmolVLA LIBERO-Spatial, seed 1001, fixtures free (cell B), rerun with the logging runner so its noise stream can be
# checked against the pinned cell (C). Same arguments as every other evaluation-seed cell.
# Usage: run_spatial_seed1001_free_logged.sh <output dir>
O=$1; mkdir -p $O
export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1 MUJOCO_GL=egl TZ=ICT-7
PY=/workspace/venvs/lerobot/bin/python; S=/workspace/libero-vla-eval/scripts
# New-pod setup outside the volume: LIBERO config (no Y/N prompt), EGL libraries, LeRobot LIBERO assets.
echo N | $PY -c "from libero.libero import get_libero_path" > /dev/null 2>&1
apt-get install -y -qq libegl1 libgl1 libglew-dev libosmesa6-dev > /dev/null 2>&1
mkdir -p /root/.cache/libero && [ -e /root/.cache/libero/assets ] || ln -s /workspace/libero_assets_hf /root/.cache/libero/assets
RESET_LOG=$O/spatial_seed1001_free_resets.jsonl SCENE_SEED_SHIFT=0 $PY $S/run_lerobot_libero_logged.py \
  --policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero --env.task=libero_spatial \
  --eval.batch_size=50 --eval.n_episodes=50 --eval.use_async_envs=false --policy.n_action_steps=1 --policy.use_amp=false \
  --policy.device=cuda --seed=1001 --output_dir=$O/eval
rc=$?
# Episode count in the same format the chain counts (one line per episode, "success": true/false).
$PY - "$O" <<'EOF'
import json, sys, os
o = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(o, "spatial_seed1001_free_resets.jsonl"))]
print(f"reset records: {len(rows)}")
EOF
$PY $S/compare_episode_logs.py $O/spatial_seed1001_free_resets.jsonl \
  /workspace/runs/pinned_spatial/spatial_seed1001_pinned_resets.jsonl > $O/compare_vs_pinned.txt 2>&1
exit $rc
