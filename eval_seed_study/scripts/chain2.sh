#!/usr/bin/env bash
set -u
cd /workspace/runs/eval_seed
export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1 MUJOCO_GL=egl TZ=ICT-7
PY=/workspace/venvs/lerobot/bin/python; S=/workspace/libero-vla-eval/scripts
P=/workspace/runs/eval_seed
note() { date "+%F %T $*" >> chain.log; curl -s -d "eval-seed cells: $*" ntfy.sh/${NTFY_TOPIC:?set NTFY_TOPIC} > /dev/null; }
cell() {  # $1 out dir, $2 name, $3 suite, $4 seed, $5 scene shift
  date "+%F %T start $2" >> chain.log
  RESET_LOG=$1/$2_resets.jsonl SCENE_SEED_SHIFT=$5 $PY $S/run_lerobot_libero_logged.py \
    --policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero --env.task=$3 \
    --eval.batch_size=50 --eval.n_episodes=50 --eval.use_async_envs=false --policy.n_action_steps=1 --policy.use_amp=false \
    --policy.device=cuda --seed=$4 --output_dir=$1/$2 > $1/$2.log 2>&1
  note "done $2 exit $?"
}
cell $P spatial_seed1001_pinned libero_spatial 1001 -1
$PY $S/compare_episode_logs.py $P/spatial_seed1001_pinned_resets.jsonl $P/spatial_1000.jsonl > $P/pin_check.txt 2>&1
cell $PWD object_seed1001_free libero_object 1001 0
cell $PWD goal_seed1001_free libero_goal 1001 0
cell $PWD goal_seed1000_floor libero_goal 1000 0
if grep -q "fixtures identical: 500/500" $P/pin_check.txt && grep -q "first frames identical: 500/500" $P/pin_check.txt; then
  cell $PWD goal_seed1001_pinned libero_goal 1001 -1
else
  note "SKIP goal pinned: Spatial pin check not 500/500, see $P/pin_check.txt"
fi
note "chain finished"
