"""Pair two LIBERO runs episode by episode (task, initial state): success with Wilson CI, flips, exact McNemar,
flips per task, first-frame check. Usage: cell_stats.py <label> <run_dir_a> <run_dir_b>
(each run dir has eval_info.json next to <run_dir>_resets.jsonl and <run_dir>.log)."""
import json, math, os, re, sys

def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

def load(run):
    info = json.load(open(f"{run}/eval_info.json"))
    succ = {(t["task_id"], i): bool(s) for t in info["per_task"] for i, s in enumerate(t["metrics"]["successes"])}
    frames = {}
    for r in map(json.loads, open(f"{run}_resets.jsonl")):
        if r.get("caller", "explicit") == "explicit":
            frames[(r["task_id"], r["init_state_idx"])] = r.get("frame_sha1") or r.get("first_frame_sha1")
    log = open(f"{run}.log", errors="ignore").read() if os.path.exists(f"{run}.log") else ""
    cfg = {}
    for k in ("seed", "batch_size", "use_amp", "n_action_steps"):
        m = re.search(rf"'{k}': ([^,}}\n]+)", log)
        cfg[k] = m.group(1) if m else "?"
    return succ, frames, cfg

label, a_dir, b_dir = sys.argv[1:4]
(sa, fa, ca), (sb, fb, cb) = load(a_dir), load(b_dir)
keys = sorted(set(sa) & set(sb))
n = len(keys)
ka, kb = sum(sa[k] for k in keys), sum(sb[k] for k in keys)
up = sum(sb[k] and not sa[k] for k in keys)
down = sum(sa[k] and not sb[k] for k in keys)
per_task = {}
for k in keys:
    if sa[k] != sb[k]:
        per_task[k[0]] = per_task.get(k[0], 0) + 1
same_frame = sum(fa.get(k) is not None and fa.get(k) == fb.get(k) for k in keys)
wa, wb = wilson(ka, n), wilson(kb, n)
print(f"== {label}")
print(f"  config A {ca} | B {cb}")
print(f"  A {ka}/{n} = {100*ka/n:.1f}% CI [{wa[0]:.1f}, {wa[1]:.1f}] | B {kb}/{n} = {100*kb/n:.1f}% CI [{wb[0]:.1f}, {wb[1]:.1f}]")
print(f"  flips {up+down}/{n} = {100*(up+down)/n:.1f}% (fail->success {up}, success->fail {down}), McNemar p = {mcnemar(up, down):.3f}")
print(f"  flips per task: {dict(sorted(per_task.items()))}")
print(f"  first frames identical: {same_frame}/{n}")
