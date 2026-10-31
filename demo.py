"""Demo: one stored pattern, one noisy probe, every rule.

Run: python demo.py
"""

import random

from comparison import (
    attention_output,
    dot,
    membrane_winner,
    modern_hopfield_output,
    noisy,
    rand_pattern,
    winner_from_output,
)

rng = random.Random(42)
n, p = 64, 16
patterns = [rand_pattern(n, rng) for _ in range(p)]
target = 5
probe = noisy(patterns[target], 0.30, rng)

winner_a, overlaps = membrane_winner(patterns, probe)
print(f"target {target}, membrane winner {winner_a}, max overlap {max(overlaps):.3f}")
for beta in (0.125, 1.0, 10.0, 50.0):
    out, weights = modern_hopfield_output(patterns, probe, beta)
    winner, _ = winner_from_output(patterns, out)
    print(f"hopfield beta={beta:<6} winner {winner}, fidelity "
          f"{abs(dot(patterns[target], out)) / (sum(x * x for x in out) ** 0.5):.3f}, "
          f"max weight {max(weights):.3f}")
out_c, weights_c, _ = attention_output(patterns, probe)
winner_c, _ = winner_from_output(patterns, out_c)
print(f"attention (beta=0.125) winner {winner_c}, max weight {max(weights_c):.3f}")
print("Note: attention must equal hopfield beta=0.125 exactly; see tests.")
