# RESULTS — membrane vs modern Hopfield vs attention

Paired trials: same patterns, same probe, every rule. N = 64.
Accuracy = winner == target. Fidelity = |<target, retrieved>|.
Attention (identity QKV) is Hopfield with beta = 1/sqrt(N) = 0.125,
so those two columns must match; if they do not, the code is wrong.

## Noisy retrieval (100 trials per cell)
### P16_s0.0
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 1.000 | 1.000 | 1.000 |
| hopfield_b0.125 | 0.080 | 0.292 | 0.070 |
| hopfield_b1 | 0.970 | 0.599 | 0.152 |
| hopfield_b10 | 1.000 | 1.000 | 0.999 |
| hopfield_b50 | 1.000 | 1.000 | 1.000 |
| attention | 0.080 | 0.292 | 0.070 |

### P16_s0.15
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 1.000 | 1.000 | 1.000 |
| hopfield_b0.125 | 0.140 | 0.282 | 0.067 |
| hopfield_b1 | 0.600 | 0.466 | 0.111 |
| hopfield_b10 | 1.000 | 0.999 | 0.935 |
| hopfield_b50 | 1.000 | 1.000 | 1.000 |
| attention | 0.140 | 0.282 | 0.067 |

### P16_s0.3
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 0.890 | 0.904 | 1.000 |
| hopfield_b0.125 | 0.070 | 0.252 | 0.065 |
| hopfield_b1 | 0.290 | 0.360 | 0.090 |
| hopfield_b10 | 0.890 | 0.904 | 0.617 |
| hopfield_b50 | 0.890 | 0.921 | 0.942 |
| attention | 0.070 | 0.252 | 0.065 |

### P64_s0.0
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 1.000 | 1.000 | 1.000 |
| hopfield_b0.125 | 0.000 | 0.150 | 0.018 |
| hopfield_b1 | 0.510 | 0.395 | 0.041 |
| hopfield_b10 | 1.000 | 1.000 | 0.994 |
| hopfield_b50 | 1.000 | 1.000 | 1.000 |
| attention | 0.000 | 0.150 | 0.018 |

### P64_s0.15
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 1.000 | 1.000 | 1.000 |
| hopfield_b0.125 | 0.030 | 0.166 | 0.017 |
| hopfield_b1 | 0.190 | 0.307 | 0.029 |
| hopfield_b10 | 1.000 | 0.995 | 0.807 |
| hopfield_b50 | 1.000 | 1.000 | 1.000 |
| attention | 0.030 | 0.166 | 0.017 |

### P64_s0.3
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 0.810 | 0.831 | 1.000 |
| hopfield_b0.125 | 0.020 | 0.153 | 0.016 |
| hopfield_b1 | 0.110 | 0.223 | 0.023 |
| hopfield_b10 | 0.840 | 0.793 | 0.338 |
| hopfield_b50 | 0.810 | 0.850 | 0.926 |
| attention | 0.020 | 0.153 | 0.016 |

### P256_s0.0
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 1.000 | 1.000 | 1.000 |
| hopfield_b0.125 | 0.000 | 0.135 | 0.004 |
| hopfield_b1 | 0.520 | 0.393 | 0.010 |
| hopfield_b10 | 1.000 | 1.000 | 0.975 |
| hopfield_b50 | 1.000 | 1.000 | 1.000 |
| attention | 0.000 | 0.135 | 0.004 |

### P256_s0.15
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 1.000 | 1.000 | 1.000 |
| hopfield_b0.125 | 0.000 | 0.127 | 0.004 |
| hopfield_b1 | 0.140 | 0.262 | 0.007 |
| hopfield_b10 | 1.000 | 0.982 | 0.549 |
| hopfield_b50 | 1.000 | 0.998 | 0.995 |
| attention | 0.000 | 0.127 | 0.004 |

### P256_s0.3
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 0.650 | 0.677 | 1.000 |
| hopfield_b0.125 | 0.000 | 0.113 | 0.004 |
| hopfield_b1 | 0.040 | 0.178 | 0.006 |
| hopfield_b10 | 0.660 | 0.604 | 0.112 |
| hopfield_b50 | 0.650 | 0.725 | 0.785 |
| attention | 0.000 | 0.113 | 0.004 |

## Occluded retrieval, 50% coordinates zeroed (100 trials)
### P16
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 1.000 | 1.000 | 1.000 |
| hopfield_b0.125 | 0.060 | 0.271 | 0.068 |
| hopfield_b1 | 0.680 | 0.478 | 0.117 |
| hopfield_b10 | 1.000 | 1.000 | 0.965 |
| hopfield_b50 | 1.000 | 1.000 | 1.000 |
| attention | 0.060 | 0.271 | 0.068 |

### P64
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 1.000 | 1.000 | 1.000 |
| hopfield_b0.125 | 0.010 | 0.151 | 0.017 |
| hopfield_b1 | 0.210 | 0.308 | 0.031 |
| hopfield_b10 | 1.000 | 0.999 | 0.881 |
| hopfield_b50 | 1.000 | 1.000 | 1.000 |
| attention | 0.010 | 0.151 | 0.017 |

## Correlated patterns, sigma = 0.15 (100 trials)
### P64_rho0.6
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 0.950 | 0.984 | 1.000 |
| hopfield_b0.125 | 0.030 | 0.786 | 0.016 |
| hopfield_b1 | 0.030 | 0.789 | 0.020 |
| hopfield_b10 | 0.560 | 0.853 | 0.142 |
| hopfield_b50 | 0.960 | 0.986 | 0.882 |
| attention | 0.030 | 0.786 | 0.016 |

### P64_rho0.9
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 0.340 | 0.940 | 1.000 |
| hopfield_b0.125 | 0.040 | 0.951 | 0.016 |
| hopfield_b1 | 0.040 | 0.951 | 0.017 |
| hopfield_b10 | 0.050 | 0.953 | 0.033 |
| hopfield_b50 | 0.290 | 0.962 | 0.250 |
| attention | 0.040 | 0.951 | 0.016 |

## Random-projection attention, P64 sigma 0.15 (50 trials)
| rule | accuracy | mean fidelity | mean max weight |
|---|---|---|---|
| membrane | 1.000 | 1.000 | 1.000 |
| hopfield_b0.125 | 0.020 | 0.191 | 0.017 |
| hopfield_b1 | 0.280 | 0.316 | 0.029 |
| hopfield_b10 | 1.000 | 0.994 | 0.774 |
| hopfield_b50 | 1.000 | 1.000 | 1.000 |
| attention | 0.020 | 0.191 | 0.017 |
| attention_random_proj | 0.020 | 0.225 | 0.016 |

## Complex sanity: classic rule, complex patterns, sigma 0.15
- P16_s0.15: accuracy 1.000 (compare with real membrane at the same cell in the noisy table)
- P64_s0.15: accuracy 1.000 (compare with real membrane at the same cell in the noisy table)

## Refusal: frozen tau, fresh test probes (200 in + 200 out)
- tau = 0.430, calibration J = 0.990
- accepted in-library: 199/200, accepted out-of-library: 2/200
- membrane: accuracy-on-accepted 1.000, coverage 0.995, false-accept rate 0.010
- hopfield_b0.125: accuracy-on-accepted 0.045, coverage 0.995, false-accept rate 0.010
- hopfield_b1: accuracy-on-accepted 0.271, coverage 0.995, false-accept rate 0.010
- hopfield_b10: accuracy-on-accepted 1.000, coverage 0.995, false-accept rate 0.010
- hopfield_b50: accuracy-on-accepted 1.000, coverage 0.995, false-accept rate 0.010
- attention: accuracy-on-accepted 0.045, coverage 0.995, false-accept rate 0.010

## How to read this
Classification accuracy answers 'did the right label win?'.
Fidelity answers 'how clean is the retrieved vector?'. A rule
can win the label while returning a blend; that is the gap
between classification capacity and recall capacity, and it is
why the two columns are reported side by side, never merged.
