"""Numbers for the final report and slides that final_analysis.py does not print (CPU only, standard library only).
1. Success rate per cell with a Wilson 95% interval.
2. Every paired comparison: flips, Wilson interval of the flip rate, direction counts, McNemar exact p.
3. The identity flips(A vs B) = flips(A vs C) + flips(C vs B) - 2 x (episodes flipped in both steps).
4. Spatial by task group (bowl in or on furniture: tasks 4, 7, 9; the other seven): flips per pair and the
   furniture term flips(A vs B) - flips(A vs C) with a paired bootstrap (20,000 draws), for SmolVLA and OpenVLA-7B.
5. Seed standard deviation of one 500-episode score, sqrt(f / 2n), and the minimum gap between two models,
   (z_0.975 + z_0.80) * sqrt((fA + fB) / 2n), from the measured reseed flip rates.
6. Ranking check: SmolVLA minus OpenVLA-7B per cell on Object and Goal (point gaps)."""
import json, math, os, random
from statistics import NormalDist

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
N = 500

def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

def outcomes(run):
    info = json.load(open(f"{R}/{run}/eval_info.json"))
    return {(t["task_id"], i): bool(s) for t in info["per_task"] for i, s in enumerate(t["metrics"]["successes"])}

def openvla(run):
    return {(r["task_id"], r["episode_idx"]): bool(r["success"]) for r in map(json.loads, open(f"{R}/{run}/episodes.jsonl"))}

CELLS = {
    "Spatial A (seed 1000)": "smolvla_spatial_seed1000",
    "Spatial B (seed 1001, logged rerun)": "smolvla_spatial_seed1001_free_logged",
    "Spatial B (seed 1001, first run)": "smolvla_spatial_seed1001_free",
    "Spatial C (noise 1001, furniture 1000)": "smolvla_spatial_seed1001_pinned",
    "Goal A (seed 1000)": "smolvla_goal_seed1000",
    "Goal A (seed 1000, same-seed rerun)": "smolvla_goal_seed1000_rerun",
    "Goal B (seed 1001)": "smolvla_goal_seed1001_free",
    "Goal C (noise 1001, furniture 1000)": "smolvla_goal_seed1001_pinned",
    "Object A (seed 1000)": "smolvla_object_seed1000",
    "Object B (seed 1001)": "smolvla_object_seed1001_free",
}
O = {name: outcomes(run) for name, run in CELLS.items()}
ov1 = openvla("openvla7b_spatial_envseed1_states0-19")
ov0_all = openvla("openvla7b_spatial_envseed0_states0-49")
ov0 = {k: ov0_all[k] for k in ov1}
O["OpenVLA-7B Spatial, env seed 0 (states 0-19)"] = ov0
O["OpenVLA-7B Spatial, env seed 1 (states 0-19)"] = ov1

print("== 1. Success per cell, Wilson 95% interval")
for name, o in O.items():
    k, n = sum(o.values()), len(o)
    lo, hi = wilson(k, n)
    print(f"  {name}: {k}/{n} = {100*k/n:.1f}% [{lo:.1f}, {hi:.1f}]")

def pair(x, y, keys=None):
    ks = keys if keys is not None else sorted(set(O[x]) & set(O[y]))
    up = sum(O[y][k] and not O[x][k] for k in ks)
    down = sum(O[x][k] and not O[y][k] for k in ks)
    return up, down, len(ks)

PAIRS = [
    ("Spatial A vs B (both)", "Spatial A (seed 1000)", "Spatial B (seed 1001, logged rerun)"),
    ("Spatial A vs C (noise only)", "Spatial A (seed 1000)", "Spatial C (noise 1001, furniture 1000)"),
    ("Spatial C vs B (furniture only)", "Spatial C (noise 1001, furniture 1000)", "Spatial B (seed 1001, logged rerun)"),
    ("Spatial A vs first B", "Spatial A (seed 1000)", "Spatial B (seed 1001, first run)"),
    ("Spatial C vs first B", "Spatial C (noise 1001, furniture 1000)", "Spatial B (seed 1001, first run)"),
    ("Spatial first B vs logged B", "Spatial B (seed 1001, first run)", "Spatial B (seed 1001, logged rerun)"),
    ("Goal A vs B (both)", "Goal A (seed 1000)", "Goal B (seed 1001)"),
    ("Goal A vs C (noise only)", "Goal A (seed 1000)", "Goal C (noise 1001, furniture 1000)"),
    ("Goal C vs B (furniture only)", "Goal C (noise 1001, furniture 1000)", "Goal B (seed 1001)"),
    ("Goal A vs same-seed rerun (floor)", "Goal A (seed 1000)", "Goal A (seed 1000, same-seed rerun)"),
    ("Object A vs B (both)", "Object A (seed 1000)", "Object B (seed 1001)"),
    ("OpenVLA-7B env seed 0 vs 1 (furniture only)", "OpenVLA-7B Spatial, env seed 0 (states 0-19)", "OpenVLA-7B Spatial, env seed 1 (states 0-19)"),
]
print("\n== 2. Paired comparisons: flips (fail->success / success->fail), Wilson 95% interval of the flip rate, McNemar exact p")
for label, x, y in PAIRS:
    up, down, n = pair(x, y)
    lo, hi = wilson(up + down, n)
    print(f"  {label}: {up+down}/{n} = {100*(up+down)/n:.1f}% [{lo:.1f}, {hi:.1f}] ({up} / {down}), p = {mcnemar(up, down):.3f}")

print("\n== 3. Identity flips(A vs B) = flips(A vs C) + flips(C vs B) - 2 x shared; shared expected if independent = flips(A vs C) x flips(C vs B) / n")
for suite, a, b, c in [("Spatial", "Spatial A (seed 1000)", "Spatial B (seed 1001, logged rerun)", "Spatial C (noise 1001, furniture 1000)"),
                       ("Goal", "Goal A (seed 1000)", "Goal B (seed 1001)", "Goal C (noise 1001, furniture 1000)")]:
    ks = sorted(O[a])
    fab = sum(O[a][k] != O[b][k] for k in ks); fac = sum(O[a][k] != O[c][k] for k in ks); fcb = sum(O[c][k] != O[b][k] for k in ks)
    shared = sum(O[a][k] != O[c][k] and O[c][k] != O[b][k] for k in ks)
    print(f"  {suite}: {fac} + {fcb} - 2 x {shared} = {fac + fcb - 2*shared} (A vs B: {fab}); shared expected {fac*fcb/len(ks):.1f}")

def boot(d, reps=20000, seed=0):
    rng = random.Random(seed)
    st = sorted(100 * sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(reps))
    return st[int(0.025 * reps)], st[int(0.975 * reps)], st[int(0.95 * reps)]

print("\n== 4. Spatial by task group; exploratory (bowl in or on furniture: tasks 4, 7, 9)")
A, B, C = "Spatial A (seed 1000)", "Spatial B (seed 1001, logged rerun)", "Spatial C (noise 1001, furniture 1000)"
for gname, grp in [("tasks 4, 7, 9", {4, 7, 9}), ("other seven tasks", set(range(10)) - {4, 7, 9})]:
    ks = [k for k in sorted(O[A]) if k[0] in grp]
    parts = []
    for lab, x, y in [("both", A, B), ("noise only", A, C), ("furniture only", C, B)]:
        up, down, n = pair(x, y, ks)
        parts.append(f"{lab} {up+down}/{n} = {100*(up+down)/n:.1f}%")
    d = [int(O[A][k] != O[B][k]) - int(O[A][k] != O[C][k]) for k in ks]
    lo, hi, u = boot(d)
    print(f"  SmolVLA, {gname}: " + "; ".join(parts) + f"; furniture term {100*sum(d)/len(d):+.1f} points [{lo:+.1f}, {hi:+.1f}]")
    oks = [k for k in sorted(ov1) if k[0] in grp]
    of = sum(ov0[k] != ov1[k] for k in oks)
    print(f"  OpenVLA-7B, {gname}: env seed 0 vs 1 flips {of}/{len(oks)} = {100*of/len(oks):.1f}%")

z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)
print(f"\n== 5. Seed SD sqrt(f/2n) and minimum gap {z:.3f} x sqrt((fA+fB)/2n), points, n = {N}")
f_openvla = sum(ov0[k] != ov1[k] for k in ov1) / len(ov1)
reseed = {"Spatial": ("Spatial A (seed 1000)", "Spatial B (seed 1001, logged rerun)"),
          "Goal": ("Goal A (seed 1000)", "Goal B (seed 1001)"),
          "Object": ("Object A (seed 1000)", "Object B (seed 1001)")}
# OpenVLA-7B's reseed flip rate: measured on Spatial; assumed equal on Goal; zero on Object (deterministic, no furniture)
f_ov = {"Spatial": f_openvla, "Goal": f_openvla, "Object": 0.0}
for suite, (a, b) in reseed.items():
    up, down, n = pair(a, b); f = (up + down) / n
    print(f"  {suite}: f = {f:.3f}; seed SD {100*math.sqrt(f/(2*N)):.2f}; minimum gap vs a model with the same f "
          f"{100*z*math.sqrt(2*f/(2*N)):.2f}; vs OpenVLA-7B (f = {f_ov[suite]:.2f}) {100*z*math.sqrt((f+f_ov[suite])/(2*N)):.2f}")

print("\n== 6. Ranking check: SmolVLA minus OpenVLA-7B, points")
# OpenVLA-7B success on the same 500 Object and Goal episodes, from the first author's evaluation pipeline
# (env seed 0, original OpenVLA script); these two runs are not part of this repository.
OPENVLA_OBJECT, OPENVLA_GOAL = 346, 388
for suite, ref, cells in [("Object", OPENVLA_OBJECT, ["Object A (seed 1000)", "Object B (seed 1001)"]),
                          ("Goal", OPENVLA_GOAL, ["Goal A (seed 1000)", "Goal B (seed 1001)", "Goal C (noise 1001, furniture 1000)",
                                                  "Goal A (seed 1000, same-seed rerun)"])]:
    gaps = [f"{name.split(' ', 1)[1]} {100*(sum(O[name].values()) - ref)/N:+.1f}" for name in cells]
    print(f"  {suite} (OpenVLA-7B {ref}/{N} = {100*ref/N:.1f}%): " + "; ".join(gaps))
