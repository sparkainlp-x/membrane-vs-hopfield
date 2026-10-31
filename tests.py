"""Tests for membrane-vs-hopfield. Run: python tests.py -v

Every test is a claim the benchmark depends on. If one fails, the
benchmark's numbers do not mean what the README says they mean.
"""

import math
import random
import unittest

from comparison import (
    attention_output,
    calibrate_tau,
    correlated_patterns,
    dot,
    membrane_winner,
    modern_hopfield_output,
    normalize,
    occluded,
    rand_pattern,
    softmax,
    winner_from_output,
)


class TestIdentities(unittest.TestCase):
    def test_attention_identity_equals_hopfield_at_beta_1_over_sqrt_n(self):
        """The Ramsauer identity, to machine precision."""
        rng = random.Random(11)
        n, p = 32, 8
        patterns = [rand_pattern(n, rng) for _ in range(p)]
        probe = rand_pattern(n, rng)
        out_b, _ = modern_hopfield_output(patterns, probe, 1.0 / math.sqrt(n))
        out_c, _, _ = attention_output(patterns, probe)
        err = max(abs(a - b) for a, b in zip(out_b, out_c))
        self.assertLess(err, 1e-12)

    def test_large_beta_hopfield_collapses_to_membrane_winner(self):
        rng = random.Random(12)
        n, p = 32, 16
        patterns = [rand_pattern(n, rng) for _ in range(p)]
        probe = patterns[3]
        winner_a, _ = membrane_winner(patterns, probe)
        out_b, _ = modern_hopfield_output(patterns, probe, 1000.0)
        winner_b, _ = winner_from_output(patterns, out_b)
        self.assertEqual(winner_a, winner_b)

    def test_softmax_sums_to_one_and_is_stable(self):
        w = softmax([1000.0, 1001.0, 999.0])
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in w))

    def test_clean_probe_is_retrieved_by_membrane_and_high_beta_only(self):
        # Real finding, not a test bug: a clean probe is retrieved by
        # argmax and by sharp Hopfield (beta >= 10), but NOT reliably
        # by diffuse Hopfield (beta = 0.125, the attention default).
        # At beta = 0.125 the softmax weights are near-uniform
        # (max weight ~0.07 at P=16), so the blended output's nearest
        # pattern by second projection is often a bystander, even
        # though the blend still contains the target. The test pins
        # that fact instead of assuming all rules retrieve clean probes.
        rng = random.Random(13)
        n, p = 64, 16
        patterns = [rand_pattern(n, rng) for _ in range(p)]
        probe = patterns[5]
        winner_a, _ = membrane_winner(patterns, probe)
        self.assertEqual(winner_a, 5)
        for beta in (10.0, 50.0):
            out, _ = modern_hopfield_output(patterns, probe, beta)
            winner, _ = winner_from_output(patterns, out)
            self.assertEqual(winner, 5)
        out, weights = modern_hopfield_output(patterns, probe, 0.125)
        self.assertLess(max(weights), 0.2)

    def test_correlated_patterns_have_about_the_target_rho(self):
        rng = random.Random(14)
        pats = correlated_patterns(64, 20, 0.6, rng)
        dots = [dot(pats[i], pats[j]) for i in range(20) for j in range(i + 1, 20)]
        mean_dot = sum(dots) / len(dots)
        self.assertGreater(mean_dot, 0.45)
        self.assertLess(mean_dot, 0.75)

    def test_occlusion_keeps_unit_norm(self):
        rng = random.Random(15)
        v = rand_pattern(64, rng)
        occ = occluded(v, rng, frac=0.5)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in occ)), 1.0)

    def test_calibrate_tau_separates_in_from_ood(self):
        rng = random.Random(16)
        in_scores = [0.9 + rng.gauss(0, 0.03) for _ in range(100)]
        ood_scores = [0.2 + rng.gauss(0, 0.05) for _ in range(100)]
        cal = calibrate_tau(in_scores, ood_scores)
        self.assertGreater(cal["J"], 0.9)
        self.assertGreater(cal["tau"], 0.4)
        self.assertLess(cal["tau"], 0.8)

    def test_membrane_winner_is_argmax_dot(self):
        patterns = [[1.0, 0.0], [0.0, 1.0]]
        probe = normalize([0.9, 0.1])
        winner, overlaps = membrane_winner(patterns, probe)
        self.assertEqual(winner, 0)
        self.assertGreater(overlaps[0], overlaps[1])


if __name__ == "__main__":
    unittest.main()
