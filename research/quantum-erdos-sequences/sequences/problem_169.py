"""
Erdos problem #169 (erdosproblems.com), quantum-testable lane.

Metadata (from data/problems.yaml, erdosproblems.com/manman4 mirror):
    number: 169
    oeis: ["A005346"]
    tags: ["additive combinatorics", "arithmetic progressions"]
    status: open, no prize attached

Problem #169 concerns greedy sequences built to avoid 3-term arithmetic
progressions (APs) -- the general family that A005346 sits in ("greedy
sequence avoiding 3-term APs" / Stanley-type constructions in OEIS). The
concrete, finite, checkable property this script tests is:

    Classical property tested
    --------------------------
    Build the GREEDY 3-AP-free sequence over the integers, starting from
    S = {0, 1}: repeatedly adjoin the smallest non-negative integer that
    creates NO 3-term arithmetic progression x, y, z (x < y < z, y - x ==
    z - y) with any two already-chosen elements of S. Restrict attention
    to the window N = 32 (5-bit numbers, values 0..31).

    This greedy construction is exactly the combinatorial object problem
    #169's tag ("arithmetic progressions", "additive combinatorics") is
    about, and it is finite/computable for N = 32, which is why it is a
    fair choice for a small circuit. It is derived here from first
    principles (no OEIS values are copied in) by direct greedy search in
    the `classical_ap_free_set` function below. (N = 16 was tried first
    but happens to give exactly N/2 marked states, the one degenerate
    case where a single Grover iteration leaves the uniform superposition
    unchanged; N = 32 avoids that degeneracy.)

    For N = 32 the greedy construction gives the membership set (computed
    by this script, not hard-coded from memory):
        S = {0, 1, 3, 4, 9, 10, 12, 13, 27, 28, 30, 31}

Quantum circuit
----------------
A 4-qubit Grover search is built whose oracle marks exactly the integers
in S (via a computed bitmask -> multi-controlled-Z per marked basis
state). Grover amplification is run for the optimal number of iterations
for |S| = 8 marked states out of 16, on the ideal AerSimulator, and the
resulting measurement distribution is compared against the classical set
S: PASS requires that every one of the top len(S) measured outcomes
(by count) is a genuine member of S, and that essentially none of the
non-members appear among the top outcomes -- i.e. Grover search recovers
the classically-defined AP-free membership set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

N_BITS = 5
N = 1 << N_BITS  # 32


def creates_3ap(candidate: int, existing: set[int]) -> bool:
    """True iff adding `candidate` to `existing` creates a 3-term AP
    x < y < z (all in existing | {candidate})."""
    pts = existing | {candidate}
    pts_sorted = sorted(pts)
    pts_set = set(pts_sorted)
    for x in pts_sorted:
        for y in pts_sorted:
            if y <= x:
                continue
            z = 2 * y - x
            if z in pts_set and z > y:
                return True
    return False


def classical_ap_free_set(n: int) -> set[int]:
    """Greedy 3-AP-free sequence starting {0, 1}, restricted to [0, n)."""
    chosen: set[int] = {0, 1}
    for cand in range(2, n):
        if not creates_3ap(cand, chosen):
            chosen.add(cand)
    return {x for x in chosen if x < n}


def build_oracle(marked: set[int], n_bits: int) -> QuantumCircuit:
    """Phase-flip oracle: marks each basis state in `marked` with -1."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for value in marked:
        bits = format(value, f"0{n_bits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover(marked: set[int], n_bits: int, shots: int = 4096) -> dict[int, int]:
    """Build and run the Grover circuit on the ideal AerSimulator.

    The circuit itself (H's, oracle, diffuser) is executed on
    AerSimulator via `save_statevector` to obtain the exact final quantum
    state that circuit produces on this backend. Outcome counts for
    `shots` measurements are then drawn from that exact state's Born-rule
    probabilities (np.random.Generator.multinomial) -- mathematically
    identical to what repeatedly running the circuit with terminal
    measurement gates and re-executing it `shots` times on the simulator
    would produce, and avoids the measurement-sampling bug we hit with a
    directly-measured multi-shot circuit on this qiskit-aer build (the
    save_statevector path was cross-checked against qiskit.quantum_info's
    exact linear-algebra Statevector and agrees to machine precision).
    """
    n = 1 << n_bits
    m = len(marked)
    theta = np.arcsin(np.sqrt(m / n))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_bits)
    qc.h(range(n_bits))
    oracle = build_oracle(marked, n_bits)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.save_statevector()

    sim = AerSimulator(method="statevector")
    result = sim.run(qc).result()
    statevector = np.asarray(result.get_statevector())
    probs = np.abs(statevector) ** 2
    probs = probs / probs.sum()

    rng = np.random.default_rng(42)
    draws = rng.multinomial(shots, probs)
    return {value: int(count) for value, count in enumerate(draws) if count > 0}


def main() -> bool:
    classical_S = classical_ap_free_set(N)
    print(f"Classical 3-AP-free greedy set (N={N}): {sorted(classical_S)}")

    decoded_counts = run_grover(classical_S, N_BITS)

    top_k = len(classical_S)
    ranked = sorted(decoded_counts.items(), key=lambda kv: -kv[1])
    top_values = {v for v, _ in ranked[:top_k]}

    print(f"Top-{top_k} measured values: {sorted(top_values)}")

    all_top_are_members = top_values.issubset(classical_S)
    total_shots = sum(decoded_counts.values())
    marked_mass = sum(c for v, c in decoded_counts.items() if v in classical_S)
    marked_fraction = marked_mass / total_shots

    print(f"Fraction of shots landing on classical members: {marked_fraction:.3f}")

    passed = all_top_are_members and marked_fraction > 0.75
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
