#!/usr/bin/env bash
# Remove the pod when the chain finishes, when no log was written for 30 min (hang), or at the 10:00 deadline.
DEADLINE=$(TZ=ICT-7 date -d "2026-10-12 10:00" +%s)
bye() { curl -s -d "eval-seed watchdog: $1, removing pod" ntfy.sh/${NTFY_TOPIC:?set NTFY_TOPIC} > /dev/null
        date "+%F %T watchdog: $1, removing pod" >> /workspace/runs/eval_seed/chain.log
        runpodctl remove pod "$RUNPOD_POD_ID" >> /workspace/runs/eval_seed/chain.log 2>&1; exit 0; }
while true; do
  now=$(date +%s)
  grep -q "chain finished" /workspace/runs/eval_seed/chain.log 2>/dev/null && { sleep 60; bye "chain finished"; }
  [ $now -ge $DEADLINE ] && bye "deadline 10:00"
  newest=$(stat -c %Y /workspace/runs/eval_seed/*.log /workspace/runs/eval_seed/*.log 2>/dev/null | sort -n | tail -1)
  [ $((now - newest)) -gt 1800 ] && bye "no log for 30 min (hang)"
  sleep 120
done
