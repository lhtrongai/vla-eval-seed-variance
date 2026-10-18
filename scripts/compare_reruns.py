"""Compare two reruns of the same checkpoint on the same LIBERO initial states.

Usage: python compare_reruns.py <run_a> <run_b>   (run dirs inside the current directory;
each has <run>/eval_info.json and <run>_resets.jsonl)
Writes episodes_paired.csv and compare.json, prints a summary; the last line is a one-line verdict.
"""
import csv
import json
import math
import sys

ANCHOR, ANCHOR_TOL = 78.8, 5.0          # SMILE, Spatial, one executed action
PUBLISHED_BAND_TOP = 1.8                # top of the published seed-spread band (points)
REF_EPISODES = 2000                     # the seed-count thresholds refer to the 2,000-episode four-suite rate


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def load(run):
    info = json.load(open(f"{run}/eval_info.json"))
    succ = {}
    for t in info["per_task"]:
        for i, s in enumerate(t["metrics"]["successes"]):
            succ[(t["task_id"], i)] = bool(s)
    explicit = {}
    for line in open(f"{run}_resets.jsonl"):
        r = json.loads(line)
        if r["caller"] != "explicit":
            continue
        key = (r["task_id"], r["sub_env"])
        assert key not in explicit, f"{run}: two explicit resets for {key}"
        assert r["sub_env"] == r["init_state_idx"], f"{run}: sub_env != init_state_idx at {key}"
        explicit[key] = r
    assert set(explicit) == set(succ), f"{run}: explicit resets do not match episodes"
    return info["overall"], succ, explicit


def main(run_a, run_b):
    oa, sa, ra = load(run_a)
    ob, sb, rb = load(run_b)
    keys = sorted(sa)
    assert keys == sorted(sb), "runs cover different episodes"
    n = len(keys)
    rows, init_mismatch, frame_mismatch, flips = [], 0, 0, 0
    for k in keys:
        init_same = ra[k]["init_state_sha1"] == rb[k]["init_state_sha1"]
        frame_same = ra[k]["first_frame_sha1"] == rb[k]["first_frame_sha1"]
        init_mismatch += not init_same
        frame_mismatch += not frame_same
        flips += sa[k] != sb[k]
        rows.append({"task_id": k[0], "init_state_idx": k[1], "init_state_sha1": ra[k]["init_state_sha1"],
                     "init_same": init_same, "first_frame_same": frame_same,
                     "success_a": int(sa[k]), "success_b": int(sb[k])})
    with open("episodes_paired.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ka, kb = sum(sa.values()), sum(sb.values())
    pa, pb = 100 * ka / n, 100 * kb / n
    flip_rate = flips / n
    r_here = 100 * math.sqrt(flip_rate / (2 * n))
    r_ref = 100 * math.sqrt(flip_rate / (2 * REF_EPISODES))

    pipeline_ok = all(abs(p - ANCHOR) <= ANCHOR_TOL for p in (pa, pb))
    hashes_ok = init_mismatch == 0 and frame_mismatch == 0
    noise_ok = 2.2 * r_ref <= PUBLISHED_BAND_TOP
    if r_ref <= 0.45:
        seeds = "5 seeds"
    elif r_ref <= 0.8:
        seeds = "8 seeds"
    else:
        seeds = "5 seeds (noise above the planned range)"

    out = {"n": n, "run_a": run_a, "run_b": run_b,
           "success_a": pa, "ci_a": wilson(ka, n), "success_b": pb, "ci_b": wilson(kb, n),
           "sec_per_episode": [oa.get("eval_ep_s"), ob.get("eval_ep_s")],
           "flips": flips, "flip_rate": flip_rate,
           "r_points_at_this_n": r_here, "r_points_at_2000": r_ref,
           "assumption": "r at 2000 assumes the Spatial flip rate holds for all four suites",
           "init_hash_mismatches": init_mismatch, "first_frame_mismatches": frame_mismatch,
           "pass_pipeline": pipeline_ok, "pass_hashes": hashes_ok, "pass_noise": noise_ok,
           "seed_count_rule": seeds}
    json.dump(out, open("compare.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    verdict = "PASS" if pipeline_ok and hashes_ok and noise_ok else "STOP"
    print(f"{verdict}: Spatial {pa:.1f}% / {pb:.1f}% (anchor {ANCHOR}); flips {flips}/{n}; "
          f"r={r_ref:.2f} pts at 2000 eps -> {seeds}; hash mismatches init {init_mismatch}, frame {frame_mismatch}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
