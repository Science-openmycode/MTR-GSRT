"""Exact rational discrete-Laplace noise for a bit-level pure-DP release."""
from __future__ import annotations

import random
from fractions import Fraction

import numpy as np


def _bernoulli_rational(probability: Fraction, rng) -> int:
    if not isinstance(probability, Fraction) or not 0 <= probability <= 1:
        raise ValueError("Bernoulli probability must be a rational in [0,1]")
    return int(rng.randrange(probability.denominator) < probability.numerator)


def _bernoulli_exp_unit(value: Fraction, rng) -> int:
    """Exact Bernoulli(exp(-value)) for rational value in [0,1]."""
    if not isinstance(value, Fraction) or not 0 <= value <= 1:
        raise ValueError("unit exponential argument must lie in [0,1]")
    k = 1
    while _bernoulli_rational(value / k, rng):
        k += 1
    return k % 2


def _bernoulli_exp(value: Fraction, rng) -> int:
    if not isinstance(value, Fraction) or value < 0:
        raise ValueError("exponential argument must be a nonnegative rational")
    while value > 1:
        if not _bernoulli_exp_unit(Fraction(1, 1), rng):
            return 0
        value -= 1
    return _bernoulli_exp_unit(value, rng)


def _geometric_exp_slow(value: Fraction, rng) -> int:
    result = 0
    while _bernoulli_exp(value, rng):
        result += 1
    return result


def _geometric_exp_fast(value: Fraction, rng) -> int:
    """Exact geometric(1-exp(-value)); algorithm of Canonne et al."""
    if not isinstance(value, Fraction) or value < 0:
        raise ValueError("geometric argument must be a nonnegative rational")
    if value == 0:
        return 0
    denominator = value.denominator
    while True:
        offset = rng.randrange(denominator)
        if _bernoulli_exp(Fraction(offset, denominator), rng):
            break
    coarse = _geometric_exp_slow(Fraction(1, 1), rng)
    return (coarse * denominator + offset) // value.numerator


def _sample_discrete_laplace(scale: Fraction, rng) -> int:
    """Exact PMF proportional to exp(-abs(z)/scale) on the integers."""
    if not isinstance(scale, Fraction) or scale <= 0:
        raise ValueError("discrete-Laplace scale must be a positive rational")
    while True:
        sign = _bernoulli_rational(Fraction(1, 2), rng)
        magnitude = _geometric_exp_fast(1 / scale, rng)
        if sign and magnitude == 0:
            continue
        return magnitude * (1 - 2 * sign)


def add_exact_discrete_laplace(
    integer_query: np.ndarray,
    *,
    epsilon_numerator: int,
    epsilon_denominator: int,
    sensitivity: int,
    rng=None,
    public_clamp_abs: int = 10**15,
) -> tuple[np.ndarray, dict]:
    """Apply bit-exact pure DP noise using rational epsilon and BigInt addition."""
    values = np.asarray(integer_query)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("certified discrete mechanism requires an integer query")
    epsilon = Fraction(int(epsilon_numerator), int(epsilon_denominator))
    if (
        epsilon <= 0
        or int(sensitivity) <= 0
        or int(public_clamp_abs) <= 0
        or int(public_clamp_abs) > np.iinfo(np.int64).max
    ):
        raise ValueError("epsilon, sensitivity, and public clamp must be positive")
    rng = random.SystemRandom() if rng is None else rng
    scale = Fraction(int(sensitivity), 1) / epsilon
    lower, upper = -int(public_clamp_abs), int(public_clamp_abs)
    output = np.empty(values.size, dtype=np.int64)
    for index, value in enumerate(values.ravel()):
        noisy = int(value) + _sample_discrete_laplace(scale, rng)
        output[index] = min(max(noisy, lower), upper)
    return output.reshape(values.shape), {
        "mechanism": "exact rational discrete Laplace (Canonne-Kamath-Steinke sampler)",
        "integer_l1_sensitivity": int(sensitivity),
        "epsilon_rational": f"{epsilon.numerator}/{epsilon.denominator}",
        "scale_rational": f"{scale.numerator}/{scale.denominator}",
        "public_clamp_abs": int(public_clamp_abs),
        "random_source": "OS SystemRandom" if isinstance(rng, random.SystemRandom) else "injected development RNG",
    }
