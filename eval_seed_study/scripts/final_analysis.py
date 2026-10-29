"""eval-seed evaluation-seed study, final CPU-only analysis (standard library only).
1. OpenVLA-7B env-seed-1 cell vs the env-seed-0 run on the same 200 episodes (states 0-19 of each task).
2. Paired bootstrap of the fixture term flips(A vs B) - flips(A vs C) for SmolVLA on Spatial and Goal, and pooled;
   Spatial also with the rerun of B that logs the noise stream (smolvla_spatial_seed1001_free_logged).
3. Noise only (A vs C), furniture only (C vs B) and both (A vs B): flips, and how many episodes flip under both the
   noise-only and the furniture-only change against the count expected if the two were independent."""
import json, math, os, random

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

# ---------- 1. OpenVLA-7B control ----------
old = {(r["task_id"], r["episode_idx"]): r for r in map(json.loads, open(f"{R}/openvla7b_spatial_envseed0_states0-49/episodes.jsonl"))}
new = {(r["task_id"], r["episode_idx"]): r for r in map(json.loads, open(f"{R}/openvla7b_spatial_envseed1_states0-19/episodes.jsonl"))}
s0 = {(r["task_id"], r["episode_idx"]): r for r in map(json.loads, open(f"{R}/openvla7b_spatial_envseed0_smoke/episodes.jsonl"))}
keys = sorted(new)
n = len(keys)
same_state = sum(old[k]["init_state_sha1"] == new[k]["init_state_sha1"] for k in keys)
ko, kn = sum(old[k]["success"] for k in keys), sum(new[k]["success"] for k in keys)
up = sum(new[k]["success"] and not old[k]["success"] for k in keys)
down = sum(old[k]["success"] and not new[k]["success"] for k in keys)
per_task = {}
for k in keys:
    if old[k]["success"] != new[k]["success"]:
        per_task[k[0]] = per_task.get(k[0], 0) + 1
print("== OpenVLA-7B Spatial, env seed 0 (original run, states 0-19) vs env seed 1")
print(f"  episodes {n}; env_seed values in new run: {sorted({r.get('env_seed') for r in new.values()})}; initial states identical: {same_state}/{n}")
wo, wn = wilson(ko, n), wilson(kn, n)
print(f"  seed 0: {ko}/{n} = {100*ko/n:.1f}% CI [{wo[0]:.1f}, {wo[1]:.1f}] | seed 1: {kn}/{n} = {100*kn/n:.1f}% CI [{wn[0]:.1f}, {wn[1]:.1f}]")
print(f"  flips {up+down}/{n} = {100*(up+down)/n:.1f}% (fail->success {up}, success->fail {down}), McNemar p = {mcnemar(up, down):.3f}")
print(f"  flips per task: {dict(sorted(per_task.items()))}")
# fixtures and frames: env seed 0 known only for the smoke states (0 and 1 of each task)
fx_diff, fr_diff, m = {}, {}, 0
for k in sorted(s0):
    if k in new:
        m += 1
        fx_diff.setdefault(k[0], 0); fr_diff.setdefault(k[0], 0)
        fx_diff[k[0]] += s0[k]["fixtures"] != new[k]["fixtures"]
        fr_diff[k[0]] += s0[k]["frame_sha1"] != new[k]["frame_sha1"]
print(f"  vs env seed 0 on {m} smoke episodes: fixtures differ per task {fx_diff}; first frames differ per task {fr_diff}")

# ---------- 2. Paired bootstrap of the fixture term ----------
def outcomes(run):
    info = json.load(open(f"{R}/{run}/eval_info.json"))
    return {(t["task_id"], i): bool(s) for t in info["per_task"] for i, s in enumerate(t["metrics"]["successes"])}

suites = {
    "Spatial": ("smolvla_spatial_seed1000", "smolvla_spatial_seed1001_free", "smolvla_spatial_seed1001_pinned"),
    "Goal": ("smolvla_goal_seed1000", "smolvla_goal_seed1001_free", "smolvla_goal_seed1001_pinned"),
    "Goal (A = same-runner floor run)": ("smolvla_goal_seed1000_rerun", "smolvla_goal_seed1001_free", "smolvla_goal_seed1001_pinned"),
    "Spatial (B = logged rerun)": ("smolvla_spatial_seed1000", "smolvla_spatial_seed1001_free_logged", "smolvla_spatial_seed1001_pinned"),
}
diffs = {}
for name, (a, b, c) in suites.items():
    A, B, C = outcomes(a), outcomes(b), outcomes(c)
    ks = sorted(set(A) & set(B) & set(C))
    diffs[name] = [int(A[k] != B[k]) - int(A[k] != C[k]) for k in ks]

def boot(groups, reps=20000, seed=0):
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        tot, cnt = 0, 0
        for d in groups:
            tot += sum(d[rng.randrange(len(d))] for _ in range(len(d))); cnt += len(d)
        stats.append(100 * tot / cnt)
    stats.sort()
    return stats[int(0.025 * reps)], stats[int(0.975 * reps)], stats[int(0.95 * reps)]

print("\n== Fixture term = flips(A vs B) - flips(A vs C), in points of the 500-episode rate; paired bootstrap over episodes, 20,000 draws")
for name, groups in [("Spatial", [diffs["Spatial"]]), ("Goal", [diffs["Goal"]]),
                     ("Goal (A = same-runner floor run)", [diffs["Goal (A = same-runner floor run)"]]),
                     ("Pooled Spatial + Goal", [diffs["Spatial"], diffs["Goal"]]),
                     ("Spatial (B = logged rerun)", [diffs["Spatial (B = logged rerun)"]]),
                     ("Pooled Spatial (B = logged rerun) + Goal", [diffs["Spatial (B = logged rerun)"], diffs["Goal"]])]:
    est = 100 * sum(sum(g) for g in groups) / sum(len(g) for g in groups)
    lo, hi, u95 = boot(groups)
    print(f"  {name}: {est:+.1f} points, 95% CI [{lo:+.1f}, {hi:+.1f}], one-sided 95% upper bound U = {u95:.1f}")

# ---------- 3. The two parts of the seed do not add up ----------
print("\n== Noise only (A vs C), furniture only (C vs B), both (A vs B): flips out of 500")
FURNITURE_TASKS = {4, 7, 9}  # Spatial tasks whose bowl sits in or on furniture: cabinet drawer, stove, cabinet top
for name in ("Spatial (B = logged rerun)", "Goal"):
    A, B, C = (outcomes(r) for r in suites[name])
    ks = sorted(set(A) & set(B) & set(C))
    noise = {k for k in ks if A[k] != C[k]}
    furniture = {k for k in ks if C[k] != B[k]}
    both = {k for k in ks if A[k] != B[k]}
    print(f"  {name}: noise only {len(noise)}, furniture only {len(furniture)}, both {len(both)}; flipped under noise only "
          f"and under furniture only: {len(noise & furniture)} (independent: {len(noise) * len(furniture) / len(ks):.1f})")
    if name.startswith("Spatial"):
        for label, tasks in (("bowl in or on furniture (tasks 4, 7, 9)", FURNITURE_TASKS),
                             ("other tasks", set(range(10)) - FURNITURE_TASKS)):
            n = sum(k[0] in tasks for k in ks)
            fu, no = sum(k[0] in tasks for k in furniture), sum(k[0] in tasks for k in noise)
            print(f"    {label}: furniture only {fu}/{n} = {100 * fu / n:.1f}%, noise only {no}/{n} = {100 * no / n:.1f}%")
