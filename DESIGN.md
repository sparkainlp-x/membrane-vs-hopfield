# DESIGN.md — membrane vs Hopfield vs attention

## Falsified designs (rejected before running)

1. **Comparing on different draws per rule.** If each rule got its
   own patterns and probes, a win could be luck of the draw. Rejected:
   paired trials, one pattern set and one probe per trial, all rules.
2. **Scoring a blend as if it were a pattern.** Modern Hopfield and
   attention return a weighted sum, not a stored pattern. Scoring
   that blend by "is it exactly a pattern" would fail it by
   construction. Rejected: winner is read off by a second projection,
   and fidelity is reported separately from accuracy.
3. **Letting beta float to whatever wins.** Beta is the whole
   difference between Hopfield and attention. Rejected: beta is
   swept on a fixed grid (0.125, 1, 10, 50), attention is pinned to
   0.125, and the best-beta column is labeled as best-beta, never
   presented as "the Hopfield number".
4. **Calibrating refusal on the test set.** Rejected: tau is
   calibrated on 200+200 calibration probes, frozen, and measured on
   200+200 fresh probes, exactly as in refusing-membrane.
5. **Using hash() as a seed.** Python randomizes string hashing per
   run, so a seed derived from hash() would make reruns differ.
   Caught during writing, fixed to an explicit counter.
6. **First-perfect tau tie-breaking.** The original refusing-membrane
   calibration picks the first tau achieving maximal Youden's J,
   which sits at the edge of the separation gap. Fixed here to the
   midpoint of the maximal-J taus, for equal margin on fresh data.
7. **Assuming a clean probe is retrieved by every rule.** False for
   diffuse Hopfield (beta = 0.125): near-uniform weights make the
   blended output's nearest pattern a bystander, even on a noiseless
   probe. The test now pins that fact; the benchmark reports it as
   a result, not as a bug.

## Kept design

- Real unit vectors, N = 64, matching the membrane-capacity setup
  except for the complex-to-real domain change, which is disclosed
  in the README and checked by the complex sanity task.
- Noise sigma 0 / 0.15 / 0.30, loads P 16 / 64 / 256, occlusion 50%,
  correlation rho 0.6 / 0.9: the same stressors the membrane series
  already measured, so the new numbers join an existing record
  instead of inventing a friendlier task.
- Refusal gate on max overlap, identical for every rule, so
  accuracy-on-accepted differences are winner differences, not
  gate differences.

## Open questions (not answered here)

- What happens with iterative Hopfield dynamics (multiple updates)
  instead of one step? Out of scope; one step is what the
  attention equivalence covers.
- What does a TRAINED head do? Out of scope; training would change
  the question from "rules" to "optimization".
- Do these gaps survive real data? Unknown. Synthetic patterns have
  no structure for a learned head to exploit and no structure for a
  membrane to fail on, so both directions of transfer are unproven.
