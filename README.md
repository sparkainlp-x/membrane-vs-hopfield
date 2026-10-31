# Membrane vs Modern Hopfield vs Attention v0.1.0

**The bridge to big tech, measured instead of claimed.**

On the SAME patterns, SAME probes, SAME seeds: does Jean's membrane
projection beat, match, or lose to a modern Hopfield network
(Ramsauer et al., 2020/2021) and to single-head transformer attention?

## The answer (measured, N = 64, 100 paired trials per cell)

Attention with identity QKV *is* a modern Hopfield update with
beta = 1/sqrt(N) = 0.125. That is a theorem, verified here to 1e-12,
and confirmed in every benchmark cell: the attention column equals
the Hopfield beta = 0.125 column exactly.

On one-step pattern identification, the default attention beta
collapses, and the membrane wins:

- Noisy retrieval, P = 64, sigma = 0.15: membrane 1.000,
  attention 0.030, Hopfield beta = 1: 0.190, beta = 10: 1.000,
  beta = 50: 1.000. The same pattern holds at P = 16 and P = 256,
  at sigma = 0 and 0.30, and under 50% occlusion.
- The reason is the blend: at beta = 0.125 the largest softmax
  weight is only 0.017 at P = 64, so the retrieved vector is a
  near-uniform mean of all patterns, and its nearest pattern is
  usually a bystander. Sharp beta restores argmax, and sharp
  Hopfield then matches the membrane almost everywhere.
- The one place sharp Hopfield does NOT match the membrane is
  correlated patterns at rho = 0.6: membrane 0.950, beta = 10:
  0.560, beta = 50: 0.960. Intermediate beta blends correlated
  neighbours and loses the label; only fully sharp beta recovers.
- At rho = 0.9 every rule fails identification (membrane 0.340,
  attention 0.040) while fidelity stays high (0.94-0.96) for all
  of them, because correlated patterns are near-identical: a blend
  is close to the target even when the winner is wrong. Fidelity
  alone would hide this failure, which is why both columns exist.
- Refusal does not rescue attention: with a frozen tau = 0.430
  (Youden J = 0.990, coverage 0.995, false-accept rate 0.010),
  accuracy-on-accepted is membrane 1.000, attention 0.045,
  Hopfield beta = 10/50: 1.000. The gate rejects out-of-library
  probes; it cannot fix a wrong winner among in-library ones.
- Untrained random QKV projections change nothing: 0.020 accuracy
  at P = 64, sigma = 0.15, vs 0.020 for identity attention.

What this does NOT prove: that transformers fail. Attention in a
transformer is not used for one-step discrete pattern
identification, its projections are trained, and its beta is not
the only temperature in the system. What it does prove, on this
task family, is narrower and still useful: if the job is "name the
stored pattern from a noisy probe in one step", argmax beats
default attention by a wide margin, and Hopfield only matches
argmax once beta is tuned sharp. Full numbers in RESULTS.md.

## Run

```bash
python tests.py -v
python demo.py
python benchmark.py   # writes results.json and RESULTS.md
```

## Files

- `comparison.py` — the three rules, refusal calibration, pattern
  generators. All definitions, no benchmark code.
- `benchmark.py` — paired trials, refusal task, complex sanity check.
- `tests.py` — identity tests (attention = Hopfield at beta =
  1/sqrt(N)), large-beta collapse to membrane, calibration sanity.
- `RESULTS.md` / `results.json` — measured numbers, generated.

## Honest limits

- Real vectors, not Jean's original complex patterns. Softmax over
  complex overlaps is undefined, so the real domain is what makes
  all three rules well-defined on the same dot product. A complex
  sanity task checks the classic rule's accuracy is unchanged
  within noise; the rule comparison itself is about rules, not
  about complex phase.
- One-step retrieval only. Iterative Hopfield dynamics, trained
  QKV projections, multi-head attention, and learned beta are all
  OUT of scope. A trained head could beat every number here; an
  untrained random-projection head is included precisely to show
  that projections are not free.
- Synthetic patterns only. No claim about any real dataset, model,
  or big-tech system follows from these numbers. What follows is
  narrower: on this synthetic task family, with these rules, these
  are the measured differences.

## License

AGPL-3.0-only.

## Credits

Authors: Jean-François Brisson and Mun Seok Lee.
Lee, author of the Digital to Wave paradigm (Zenodo 21947036,
21947108), is the inspiration friend behind this wave and
membrane work. This package's code was built by Brisson with
AI assistance, except where a file already credits Lee as
author; the joint credit here is for the shared programme,
not a claim that Lee wrote every line of this package.
