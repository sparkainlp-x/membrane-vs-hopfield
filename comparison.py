"""Head-to-head retrieval rules, same patterns, same probes, same seeds.

Question: on the SAME task, does Jean's membrane projection beat, match,
or lose to (a) a modern Hopfield network (Ramsauer et al., 2020/2021)
and (b) single-head transformer attention?

Definitions, stated exactly so the comparison is falsifiable:

  Patterns: real unit vectors in R^N, i.i.d. Gaussian, normalized.
  Probe: a stored pattern + Gaussian noise (sigma), renormalized;
         or occluded (half the coordinates zeroed, renormalized);
         or from a correlated family (see correlated_patterns).
  Overlap: dot product <p_mu, probe>.

  A. MEMBRANE-PROJECT (Jean, membrane-capacity / refusing-membrane):
     winner = argmax_mu <p_mu, probe>. One-shot classification.
     This is the rule whose capacity was measured as P90 > 1024 at
     sigma = 0.15 and P90 = 4 at sigma = 0.30 (N = 64, complex
     version). Classification capacity is NOT Hopfield recall
     capacity: the rule returns a label, not a cleaned-up vector.

  B. MODERN HOPFIELD (Ramsauer et al.):
     x_new = sum_mu softmax(beta * <p_mu, probe>) * p_mu.
     Winner = argmax_mu <p_mu, x_new>. Beta is a free inverse
     temperature. Large beta -> the softmax sharpens to argmax and
     B collapses to A. Small beta -> x_new is a blend of many
     patterns, and the winner must be read off by a second projection.

  C. TRANSFORMER ATTENTION, single head, identity QKV:
     Attention = sum_mu softmax(<p_mu, probe> / sqrt(d_k)) * p_mu,
     with d_k = N. This is EXACTLY B with beta = 1/sqrt(N).
     That identity is a theorem in the Ramsauer paper, not an
     empirical claim: it is pinned by test here to machine precision.
     A head with random QKV projections is a fourth variant, D,
     included to show what projections cost when they are untrained.

  REFUSAL (applied identically to every rule):
     accept iff max_mu <p_mu, probe> >= tau, where tau is calibrated
     once on labeled calibration probes by maximizing Youden's J
     (TPR - FPR), then frozen, and all metrics are measured on FRESH
     test probes. Refused probes get no winner. Accuracy-on-accepted
     is measured only over accepted probes.

Why real vectors, when Jean's original membrane is complex?
  The complex overlap <p|probe> is complex, and softmax over complex
  numbers is undefined, so a modern Hopfield / attention rule cannot
  be written on complex overlaps without an extra, arbitrary choice
  (real part? magnitude?). The real domain makes all three rules
  well-defined on the same dot product, so any difference measured
  here is a difference of RULES, not of domain tricks. The cost is
  stated as a limit: this comparison is about retrieval rules. A
  separate sanity task checks that the complex classic rule and the
  real classic rule agree on accuracy within noise on the same
  (sigma, P) cells, so the domain change does not smuggle in a win.

Everything is stdlib only, seeded, deterministic.
"""

import math
import random


def normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return list(v)
    return [x / n for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    tot = sum(exps)
    return [e / tot for e in exps]


def rand_pattern(n, rng):
    return normalize([rng.gauss(0, 1) for _ in range(n)])


def rand_pattern_complex(n, rng):
    return normalize_complex(
        [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(n)]
    )


def normalize_complex(v):
    n = math.sqrt(sum(abs(c) ** 2 for c in v))
    return [c / n for c in v]


def overlap_complex(field, pattern):
    return sum(p.conjugate() * f for p, f in zip(pattern, field))


def noisy(state, sigma, rng):
    return normalize([x + rng.gauss(0, sigma) for x in state])


def noisy_complex(state, sigma, rng):
    return normalize_complex(
        [c + complex(rng.gauss(0, sigma), rng.gauss(0, sigma)) for c in state]
    )


def occluded(state, rng, frac=0.5):
    """Zero a random half of the coordinates, renormalize."""
    idx = list(range(len(state)))
    rng.shuffle(idx)
    kill = set(idx[: int(frac * len(state))])
    return normalize([0.0 if i in kill else x for i, x in enumerate(state)])


def correlated_patterns(n, p, rho, rng):
    """Patterns with pairwise correlation ~rho: shared + private parts."""
    common = rand_pattern(n, rng)
    out = []
    for _ in range(p):
        noise = rand_pattern(n, rng)
        out.append(
            normalize(
                [math.sqrt(rho) * c + math.sqrt(1.0 - rho) * x
                 for c, x in zip(common, noise)]
            )
        )
    return out


def membrane_winner(patterns, probe):
    """Rule A: argmax dot. Returns (winner, overlaps)."""
    overlaps = [dot(p, probe) for p in patterns]
    winner = max(range(len(overlaps)), key=lambda i: overlaps[i])
    return winner, overlaps


def modern_hopfield_output(patterns, probe, beta):
    """Rule B: one Ramsauer update. Returns (output, weights)."""
    overlaps = [dot(p, probe) for p in patterns]
    weights = softmax([beta * o for o in overlaps])
    n = len(patterns[0])
    out = [0.0] * n
    for w, p in zip(weights, patterns):
        for j in range(n):
            out[j] += w * p[j]
    return out, weights


def attention_output(patterns, probe, wq=None, wk=None, wv=None):
    """Rule C (identity QKV) or D (given projections).

    With wq = wk = wv = identity, this is exactly rule B with
    beta = 1/sqrt(N). With random projections, queries, keys and
    values live in projected spaces; the winner is then read off
    against the projected (value) patterns.
    """
    n = len(patterns[0])
    ident = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    wq = ident if wq is None else wq
    wk = ident if wk is None else wk
    wv = ident if wv is None else wv

    def proj(mat, vec):
        return [dot(row, vec) for row in mat]

    q = proj(wq, probe)
    keys = [proj(wk, p) for p in patterns]
    values = [proj(wv, p) for p in patterns]
    scores = [dot(k, q) / math.sqrt(n) for k in keys]
    weights = softmax(scores)
    out = [0.0] * n
    for w, v in zip(weights, values):
        for j in range(n):
            out[j] += w * v[j]
    return out, weights, values


def winner_from_output(patterns, output):
    overlaps = [dot(p, output) for p in patterns]
    return max(range(len(overlaps)), key=lambda i: overlaps[i]), overlaps


def random_projection(n, rng):
    """Gaussian matrix scaled 1/sqrt(N): preserves norms in expectation."""
    s = 1.0 / math.sqrt(n)
    return [[rng.gauss(0, s) for _ in range(n)] for _ in range(n)]


def calibrate_tau(in_scores, ood_scores, grid=101):
    """Maximize Youden's J over tau in [0, 1]. Returns (tau, J, tpr, fpr).

    Tie-breaking matters: many taus can achieve the same maximal J
    (any tau in the separation gap gives perfect separation). Picking
    the FIRST such tau puts the threshold at the edge of the gap,
    closest to the out-of-library distribution, which is the least
    robust choice on fresh data. This function therefore picks the
    MIDPOINT of the taus achieving maximal J: equal margin on both
    sides. That is a deliberate improvement over the first-best rule
    in the original refusing-membrane code, and it is pinned by test.
    """
    scored = []
    for g in range(grid):
        tau = g / (grid - 1)
        tpr = sum(s >= tau for s in in_scores) / len(in_scores)
        fpr = sum(s >= tau for s in ood_scores) / len(ood_scores)
        scored.append((tpr - fpr, tau, tpr, fpr))
    best_j = max(s[0] for s in scored)
    winners = [s for s in scored if abs(s[0] - best_j) <= 1e-12]
    chosen = winners[len(winners) // 2]
    return {"tau": chosen[1], "J": chosen[0], "cal_tpr": chosen[2], "cal_fpr": chosen[3]}


def evaluate_trial(patterns, probe, target, betas, proj_mats=None):
    """One paired trial: every rule sees the same patterns and probe.

    Returns a dict of per-rule results: winner-correct (bool),
    retrieval fidelity |<target-pattern, output>| for continuous
    rules (membrane-project returns the winning pattern itself, so
    its fidelity is 1.0 iff the winner is correct, else the overlap
    between the wrong winner and the target), max overlap, and the
    largest softmax weight (a confidence proxy for B/C/D).
    """
    results = {}
    winner_a, overlaps_a = membrane_winner(patterns, probe)
    results["membrane"] = {
        "correct": winner_a == target,
        "fidelity": 1.0 if winner_a == target else dot(patterns[winner_a], patterns[target]),
        "max_overlap": max(overlaps_a),
        "max_weight": 1.0,  # argmax puts all weight on one pattern
    }
    for beta in betas:
        out, weights = modern_hopfield_output(patterns, probe, beta)
        winner_b, _ = winner_from_output(patterns, out)
        results[f"hopfield_b{beta:g}"] = {
            "correct": winner_b == target,
            "fidelity": abs(dot(patterns[target], normalize(out))) if dot(out, out) > 0 else 0.0,
            "max_overlap": max(overlaps_a),
            "max_weight": max(weights),
        }
    out_c, weights_c, _ = attention_output(patterns, probe)
    winner_c, _ = winner_from_output(patterns, out_c)
    results["attention"] = {
        "correct": winner_c == target,
        "fidelity": abs(dot(patterns[target], normalize(out_c))) if dot(out_c, out_c) > 0 else 0.0,
        "max_overlap": max(overlaps_a),
        "max_weight": max(weights_c),
    }
    if proj_mats is not None:
        wq, wk, wv = proj_mats
        out_d, weights_d, values = attention_output(patterns, probe, wq, wk, wv)
        winner_d, _ = winner_from_output(values, out_d)
        results["attention_random_proj"] = {
            "correct": winner_d == target,
            "fidelity": abs(dot(values[target], normalize(out_d))) if dot(out_d, out_d) > 0 else 0.0,
            "max_overlap": max(overlaps_a),
            "max_weight": max(weights_d),
        }
    return results
