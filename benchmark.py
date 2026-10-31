"""Run the full head-to-head and write results.json + RESULTS.md.

    python benchmark.py

Paired design: within one trial, every rule sees the SAME patterns and
the SAME probe. Differences between rules therefore cannot be blamed
on different random draws. Seeds are derived from (task, P, sigma,
trial) so a rerun reproduces the same numbers exactly.

Runtime: a few minutes in pure Python. The random-projection variant
is run only on the small cell (P=64, sigma=0.15) because projecting
every pattern through an N x N matrix per trial is O(P*N^2).
"""

import json
import math
import random

from comparison import (
    attention_output,
    calibrate_tau,
    correlated_patterns,
    dot,
    evaluate_trial,
    membrane_winner,
    noisy,
    noisy_complex,
    normalize,
    occluded,
    overlap_complex,
    rand_pattern,
    rand_pattern_complex,
    random_projection,
)

N = 64
BETAS = [0.125, 1.0, 10.0, 50.0]
RULES = ["membrane", "hopfield_b0.125", "hopfield_b1", "hopfield_b10",
         "hopfield_b50", "attention", "attention_random_proj"]


def aggregate(trials_results):
    """Mean correct / fidelity per rule over trials that contain it."""
    out = {}
    for rule in RULES:
        vals = [t[rule] for t in trials_results if rule in t]
        if not vals:
            continue
        out[rule] = {
            "accuracy": sum(v["correct"] for v in vals) / len(vals),
            "mean_fidelity": sum(v["fidelity"] for v in vals) / len(vals),
            "mean_max_weight": sum(v["max_weight"] for v in vals) / len(vals),
            "n": len(vals),
        }
    return out


def run_noisy_cell(p, sigma, trials, seed_base):
    res = []
    for t in range(trials):
        rng = random.Random(seed_base + t)
        patterns = [rand_pattern(N, rng) for _ in range(p)]
        target = rng.randrange(p)
        probe = noisy(patterns[target], sigma, rng) if sigma > 0 else patterns[target]
        res.append(evaluate_trial(patterns, probe, target, BETAS))
    return aggregate(res)


def run_occluded_cell(p, trials, seed_base):
    res = []
    for t in range(trials):
        rng = random.Random(seed_base + t)
        patterns = [rand_pattern(N, rng) for _ in range(p)]
        target = rng.randrange(p)
        probe = occluded(patterns[target], rng, frac=0.5)
        res.append(evaluate_trial(patterns, probe, target, BETAS))
    return aggregate(res)


def run_correlated_cell(p, rho, trials, seed_base):
    res = []
    for t in range(trials):
        rng = random.Random(seed_base + t)
        patterns = correlated_patterns(N, p, rho, rng)
        target = rng.randrange(p)
        probe = noisy(patterns[target], 0.15, rng)
        res.append(evaluate_trial(patterns, probe, target, BETAS))
    return aggregate(res)


def run_random_proj_cell(p, sigma, trials, seed_base):
    rng_proj = random.Random(999)
    proj = (random_projection(N, rng_proj), random_projection(N, rng_proj),
            random_projection(N, rng_proj))
    res = []
    for t in range(trials):
        rng = random.Random(seed_base + t)
        patterns = [rand_pattern(N, rng) for _ in range(p)]
        target = rng.randrange(p)
        probe = noisy(patterns[target], sigma, rng)
        res.append(evaluate_trial(patterns, probe, target, BETAS, proj_mats=proj))
    return aggregate(res)


def run_complex_sanity(p, sigma, trials, seed_base):
    """Complex classic rule accuracy, for the domain-change check."""
    ok = 0
    for t in range(trials):
        rng = random.Random(seed_base + t)
        patterns = [rand_pattern_complex(N, rng) for _ in range(p)]
        target = rng.randrange(p)
        probe = noisy_complex(patterns[target], sigma, rng)
        overlaps = [abs(overlap_complex(probe, pat)) for pat in patterns]
        winner = max(range(p), key=lambda i: overlaps[i])
        ok += winner == target
    return ok / trials


def run_refusal(p, sigma, seed=42):
    """Frozen tau, fresh test probes, same gate for every rule."""
    rng = random.Random(seed)
    patterns = [rand_pattern(N, rng) for _ in range(p)]

    def in_probe(r):
        target = r.randrange(p)
        return noisy(patterns[target], sigma, r), target

    def ood_probe(r):
        return rand_pattern(N, r), None

    # Calibration set
    rng_cal = random.Random(seed + 1000)
    in_scores, ood_scores = [], []
    for _ in range(200):
        probe, _ = in_probe(rng_cal)
        _, overlaps = membrane_winner(patterns, probe)
        in_scores.append(max(overlaps))
    for _ in range(200):
        probe, _ = ood_probe(rng_cal)
        _, overlaps = membrane_winner(patterns, probe)
        ood_scores.append(max(overlaps))
    cal = calibrate_tau(in_scores, ood_scores)
    tau = cal["tau"]

    # Fresh test set
    rng_test = random.Random(seed + 2000)
    test = []
    for _ in range(200):
        probe, target = in_probe(rng_test)
        test.append((probe, target, True))
    for _ in range(200):
        probe, _ = ood_probe(rng_test)
        test.append((probe, None, False))

    per_rule = {r: {"accepted": 0, "correct_on_accepted": 0} for r in RULES if r != "attention_random_proj"}
    accepted_in = accepted_ood = 0
    for probe, target, is_in in test:
        trial = evaluate_trial(patterns, probe, target if target is not None else 0, BETAS)
        _, overlaps = membrane_winner(patterns, probe)
        if max(overlaps) < tau:
            continue
        if is_in:
            accepted_in += 1
        else:
            accepted_ood += 1
        for rule in per_rule:
            per_rule[rule]["accepted"] += 1
            if is_in and trial[rule]["correct"]:
                per_rule[rule]["correct_on_accepted"] += 1
    out = {
        "tau": tau, "calibration": cal,
        "test_accept_in": accepted_in, "test_accept_ood": accepted_ood,
        "test_n_in": 200, "test_n_ood": 200,
    }
    for rule, v in per_rule.items():
        # Accuracy-on-accepted counts only accepted IN probes as
        # correct; accepted OOD probes are false accepts, not wins.
        in_accepted = accepted_in  # same gate for all rules
        out[rule] = {
            "accuracy_on_accepted": v["correct_on_accepted"] / in_accepted if in_accepted else None,
            "coverage_in": accepted_in / 200,
            "false_accept_rate": accepted_ood / 200,
        }
    return out


def main():
    results = {"N": N, "betas": BETAS, "noisy": {}, "occluded": {},
               "correlated": {}, "random_proj": {}, "complex_sanity": {},
               "refusal": {}}
    print("noisy cells ...", flush=True)
    cell_idx = 0
    for p in (16, 64, 256):
        for sigma in (0.0, 0.15, 0.30):
            key = f"P{p}_s{sigma}"
            # Stable seed base: Python's hash() is randomized per run,
            # so it must never be used as a seed. See tests.
            results["noisy"][key] = run_noisy_cell(p, sigma, 100, 10000 + cell_idx * 1000)
            cell_idx += 1
            print(" ", key, "done", flush=True)
    print("occluded ...", flush=True)
    for p in (16, 64):
        results["occluded"][f"P{p}"] = run_occluded_cell(p, 100, 50000 + p)
    print("correlated ...", flush=True)
    for rho in (0.6, 0.9):
        results["correlated"][f"P64_rho{rho}"] = run_correlated_cell(64, rho, 100, 60000 + int(rho * 100))
    print("random projection ...", flush=True)
    results["random_proj"]["P64_s0.15"] = run_random_proj_cell(64, 0.15, 50, 70000)
    print("complex sanity ...", flush=True)
    for p in (16, 64):
        results["complex_sanity"][f"P{p}_s0.15"] = run_complex_sanity(p, 0.15, 100, 80000 + p)
    print("refusal ...", flush=True)
    results["refusal"] = run_refusal(64, 0.15)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    write_results_md(results)
    print("wrote results.json and RESULTS.md")


def fmt_table(agg, rules):
    lines = ["| rule | accuracy | mean fidelity | mean max weight |",
             "|---|---|---|---|"]
    for rule in rules:
        if rule not in agg:
            continue
        v = agg[rule]
        lines.append(f"| {rule} | {v['accuracy']:.3f} | {v['mean_fidelity']:.3f} | {v['mean_max_weight']:.3f} |")
    return "\n".join(lines)


def write_results_md(r):
    core = ["membrane", "hopfield_b0.125", "hopfield_b1", "hopfield_b10",
            "hopfield_b50", "attention"]
    parts = ["# RESULTS — membrane vs modern Hopfield vs attention",
             "",
             "Paired trials: same patterns, same probe, every rule. N = 64.",
             "Accuracy = winner == target. Fidelity = |<target, retrieved>|.",
             "Attention (identity QKV) is Hopfield with beta = 1/sqrt(N) = 0.125,",
             "so those two columns must match; if they do not, the code is wrong.",
             ""]
    parts.append("## Noisy retrieval (100 trials per cell)")
    for key, agg in r["noisy"].items():
        parts.append(f"### {key}")
        parts.append(fmt_table(agg, core))
        parts.append("")
    parts.append("## Occluded retrieval, 50% coordinates zeroed (100 trials)")
    for key, agg in r["occluded"].items():
        parts.append(f"### {key}")
        parts.append(fmt_table(agg, core))
        parts.append("")
    parts.append("## Correlated patterns, sigma = 0.15 (100 trials)")
    for key, agg in r["correlated"].items():
        parts.append(f"### {key}")
        parts.append(fmt_table(agg, core))
        parts.append("")
    parts.append("## Random-projection attention, P64 sigma 0.15 (50 trials)")
    parts.append(fmt_table(r["random_proj"]["P64_s0.15"], core + ["attention_random_proj"]))
    parts.append("")
    parts.append("## Complex sanity: classic rule, complex patterns, sigma 0.15")
    for key, val in r["complex_sanity"].items():
        parts.append(f"- {key}: accuracy {val:.3f} (compare with real membrane at the same cell in the noisy table)")
    parts.append("")
    ref = r["refusal"]
    parts.append("## Refusal: frozen tau, fresh test probes (200 in + 200 out)")
    parts.append(f"- tau = {ref['tau']:.3f}, calibration J = {ref['calibration']['J']:.3f}")
    parts.append(f"- accepted in-library: {ref['test_accept_in']}/200, accepted out-of-library: {ref['test_accept_ood']}/200")
    for rule in core:
        v = ref[rule]
        acc = v['accuracy_on_accepted']
        acc_s = f"{acc:.3f}" if acc is not None else "None"
        parts.append(f"- {rule}: accuracy-on-accepted {acc_s}, "
                     f"coverage {v['coverage_in']:.3f}, false-accept rate {v['false_accept_rate']:.3f}")
    parts.append("")
    parts.append("## How to read this")
    parts.append("Classification accuracy answers 'did the right label win?'.")
    parts.append("Fidelity answers 'how clean is the retrieved vector?'. A rule")
    parts.append("can win the label while returning a blend; that is the gap")
    parts.append("between classification capacity and recall capacity, and it is")
    parts.append("why the two columns are reported side by side, never merged.")
    with open("RESULTS.md", "w") as f:
        f.write("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
