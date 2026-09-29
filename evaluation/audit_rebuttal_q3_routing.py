"""Recompute a small, exact routing example against the published decoder.

This is a public-graph calculation. It touches no private trajectory and does
not itself establish the privacy guarantee of the preceding measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "generation" / "mtr_gsrt"
SOURCE_DIR = RELEASE / "src" / "mtr" / "DP_GSRT" / "final"
sys.path[:0] = [str(RELEASE), str(SOURCE_DIR)]

from quotient_rsp_bridge_matched import RSPBridge, portal_information_projection  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_example() -> dict:
    # Two s->d routes: s->a->d has cost 2; s->d has cost 3.
    reference = np.array([[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    cost = np.array([[0.0, 1.0, 3.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    betas = (0.1, 0.5, 1.0)
    bridge = RSPBridge(reference, cost, betas)
    rows = []
    for beta in betas:
        solution = bridge.solve(2, beta)
        z = 0.5 * math.exp(-2 * beta) + 0.5 * math.exp(-3 * beta)
        p_via = 0.5 * math.exp(-2 * beta) / z
        analytic_expectation = 2 * p_via + 3 * (1 - p_via)
        observed_expectation = float(solution.expected_cost[0])
        observed_p_via = float(solution.transition[0, 1])
        if not math.isclose(observed_expectation, analytic_expectation, abs_tol=1e-10):
            raise AssertionError("RSP expected cost differs from the path-partition calculation")
        if not math.isclose(observed_p_via, p_via, abs_tol=1e-10):
            raise AssertionError("Doob transition differs from the path Gibbs probability")
        rows.append({"beta": beta, "expected_cost": observed_expectation,
                     "probability_via_middle": observed_p_via,
                     "desirability_residual": solution.residual})

    target = 2.4
    selected = bridge.choose_beta(0, 2, target)
    expected_selected = min(rows, key=lambda row: abs(row["expected_cost"] - target))["beta"]
    if selected != expected_selected:
        raise AssertionError("Length-conditioned beta choice is not the minimum error candidate")

    public_portal_prior = np.array([0.8, 0.2])
    released_portal_mass = np.array([6.0, 4.0])
    observed_portal = portal_information_projection(public_portal_prior, released_portal_mass,
                                                    conditioned_weight=0.5, likelihood_ratio_cap=2.0)
    uniform = np.array([0.5, 0.5])
    reward = np.clip(np.log(released_portal_mass / released_portal_mass.sum()) - np.log(uniform),
                     -math.log(2.0), math.log(2.0))
    expected_portal = public_portal_prior * np.exp(0.5 * reward)
    expected_portal /= expected_portal.sum()
    if not np.allclose(observed_portal, expected_portal, atol=1e-10):
        raise AssertionError("Portal projection differs from the decoder expression")
    return {
        "classification": "PUBLIC_TOY_ROUTING_EXAMPLE_NO_PRIVATE_DATA",
        "source_sha256": sha256_file(SOURCE_DIR / "quotient_rsp_bridge_matched.py"),
        "two_path_routing": rows,
        "target_cost": target,
        "chosen_beta": selected,
        "portal_public_prior": public_portal_prior.tolist(),
        "portal_released_mass_example": released_portal_mass.tolist(),
        "portal_projected_probability": observed_portal.tolist(),
        "portal_reward_reference": "uniform, not the public geometric prior",
        "note": "The actual portal-fiber decoder uses Graph-flow for the region bridge and applies q5 at portal lifting.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New JSON result path; relative to this repository root")
    args = parser.parse_args()
    output = Path(args.out)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(run_example(), handle, indent=2)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
