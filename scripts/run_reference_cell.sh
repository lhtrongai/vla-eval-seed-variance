#!/usr/bin/env bash
# Reference cell: released SmolVLA on LIBERO-Spatial twice (same 500 states, evaluation seeds 1000 and 1001),
# compare, notify; terminate the pod only on success (keep it for diagnosis on failure).
set -u
cd /workspace/runs/dpl_ref_spatial
export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1 MUJOCO_GL=egl TZ=ICT-7
PY=/workspace/venvs/lerobot/bin/python
echo "time cgroup_bytes cgroup_max used_mb total_mb" >> mem.log
( while true; do
    echo "$(date +%T) $(cat /sys/fs/cgroup/memory.current 2>/dev/null) $(cat /sys/fs/cgroup/memory.max 2>/dev/null) $(free -m | awk '/Mem:/{print $3, $2}')" >> mem.log
    sleep 10
  done ) &
MON=$!
status=ok
for SEED in 1000 1001; do
  NAME=spatial_a1_500_seed$SEED
  date "+%F %T start $NAME" >> chain.log
  RESET_LOG=${NAME}_resets.jsonl $PY /workspace/dpl_seeds/run_lerobot_libero_hashed.py \
    --policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero --env.task=libero_spatial \
    --eval.batch_size=50 --eval.n_episodes=50 --eval.use_async_envs=false --policy.n_action_steps=1 --policy.use_amp=false \
    --policy.device=cuda --seed=$SEED --output_dir=/workspace/runs/dpl_ref_spatial/$NAME > $NAME.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then status="failed at $NAME (exit $rc)"; break; fi
  date "+%F %T done $NAME" >> chain.log
done
kill $MON
if [ "$status" = ok ]; then
  $PY /workspace/dpl_seeds/compare_reruns.py spatial_a1_500_seed1000 spatial_a1_500_seed1001 > compare.txt 2>&1 \
    || status="compare failed"
fi
MSG="dpl-seeds reference cell: $status. $(tail -1 compare.txt 2>/dev/null) peak $(sort -k4 -n mem.log | tail -1 | awk '{print $4"/"$5" MB"}')"
curl -s -d "$MSG" "ntfy.sh/${NTFY_TOPIC:?set NTFY_TOPIC}" > /dev/null
date "+%F %T $MSG" >> chain.log
if [ "$status" = ok ]; then runpodctl remove pod "$RUNPOD_POD_ID" >> chain.log 2>&1; fi
