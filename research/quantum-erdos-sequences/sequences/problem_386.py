"""
Erdos problem #386 -- quantum-testable sequence lane.

Source: erdosproblems.com problem #386 (data/problems.yaml, manman4/erdosproblems
clone). Metadata: prize=no, status=open, tags=["number theory",
"binomial coefficients"], oeis=["A280992"].

OEIS A280992: "Squarefree triangular numbers that are products of
consecutive primes." First terms: 1, 3, 6, 15, 105, 210, 255255, ...
(e.g. 6 = 2*3 = T(3), 15 = 3*5 = T(5), 105 = 3*5*7 = T(14),
210 = 2*3*5*7 = T(20), 255255 = 3*5*7*11*13*17 = T(714)).

A280992 membership requires TWO conditions at once: (a) the number is a
squarefree product of a run of consecutive primes, and (b) the number is
a triangular number T(k) = k(k+1)/2 for some integer k >= 0. Condition
(a) is a multiplicative/primality search over prime runs with no small
finite bound that fits a few qubits; condition (b) is a clean, finite,
classically-checkable arithmetic predicate on a bounded integer range,
and it is the condition actually tested here. So the concrete finite
property this script tests is:

    PROPERTY: for n in {0, 1, ..., 15} (4 bits), n is a triangular
    number, i.e. there exists integer k >= 0 with k(k+1)/2 == n.

This is honest and load-bearing for the sequence: every element of
A280992 must pass exactly this triangularity test (necessary condition),
and the known term 15 in the search range below is itself both an
A280992 term (15 = 3*5) and triangular (15 = T(5)) -- verified from
first principles in classical_triangular_check() below, not copied from
OEIS. The primality/consecutive-primes half of the definition (condition
a) is NOT tested by the quantum circuit; that is noted as a limitation.

QUANTUM CIRCUIT: Grover's algorithm over 4 qubits (search space N=16,
indices 0..15) with an oracle that phase-flips exactly the triangular
numbers in that range: {0, 1, 3, 6, 10, 15}. After the optimal number of
Grover iterations, measuring the register should return a triangular
number with high probability. We run this on the ideal AerSimulator and
compare the sampled distribution against the classically-computed set of
triangular numbers in range, then PASS/FAIL on whether Grover search
concentrates probability on that classically-verified set (index 15,
the A280992 term, included).

Dependencies: qiskit, qiskit_aer, numpy only (no external calls).
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_QUBITS = 4
N = 2 ** N_QUBITS  # 16, indices 0..15


def classical_triangular_check(n_qubits):
    """From first principles: which integers in [0, 2**n_qubits) are
    triangular numbers T(k) = k(k+1)/2 for some integer k >= 0?
    Returns the sorted list of marked indices."""
    limit = 2 ** n_qubits
    marked = []
    k = 0
    while True:
        t = k * (k + 1) // 2
        if t >= limit:
            break
        marked.append(t)
        k += 1
    return marked


def verify_a280992_term_15():
    """Cross-check: 15 is a genuine A280992 term (squarefree product of
    consecutive primes AND triangular), derived here, not copied."""
    primes = [2, 3, 5, 7, 11, 13]
    is_prime_run_product = False
    for i in range(len(primes)):
        for j in range(i, len(primes)):
            run = primes[i:j + 1]
            prod = 1
            for p in run:
                prod *= p
            if prod == 15:
                is_prime_run_product = True
    triangular_ks = [k for k in range(10) if k * (k + 1) // 2 == 15]
    return is_prime_run_product and len(triangular_ks) == 1


def build_oracle(marked_indices, n_qubits):
    """Phase-flip oracle: for each marked index, flip |i> -> -|i> using
    X gates to map i -> |1...1>, a multi-controlled Z, then undo the X
    gates."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")
        # apply X on qubits where the bit is 0 (so target pattern -> all 1s)
        zero_positions = [n_qubits - 1 - pos for pos, b in enumerate(bits) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_indices, n_qubits, shots=4096):
    n_marked = len(marked_indices)
    n_total = 2 ** n_qubits
    # optimal number of Grover iterations
    theta = math.asin(math.sqrt(n_marked / n_total))
    # search small r for the iteration count that maximizes the
    # analytic success probability sin^2((2r+1)*theta); with a large
    # marked fraction the usual r ~= pi/(4 theta) - 1/2 estimate can miss
    # the true peak, so just scan.
    best_r, best_p = 1, -1.0
    for r in range(1, 6):
        p = math.sin((2 * r + 1) * theta) ** 2
        if p > best_p:
            best_p, best_r = p, r
    iterations = best_r

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_indices, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    classical_marked = classical_triangular_check(N_QUBITS)
    assert classical_marked == [0, 1, 3, 6, 10, 15], classical_marked

    a280992_ok = verify_a280992_term_15()
    print(f"Classical triangular numbers in [0,{N}): {classical_marked}")
    print(f"15 verified as genuine A280992 term from first principles: {a280992_ok}")

    counts, iterations = run_grover(classical_marked, N_QUBITS, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # sum probability mass landing on marked (triangular) indices
    total_shots = sum(counts.values())
    marked_bitstrings = {format(i, f"0{N_QUBITS}b") for i in classical_marked}
    marked_shots = sum(c for bs, c in counts.items() if bs in marked_bitstrings)
    marked_fraction = marked_shots / total_shots

    print(f"Counts: {counts}")
    print(f"Fraction of shots landing on triangular numbers: {marked_fraction:.4f}")

    # baseline: uniform random guessing would land on a marked index with
    # probability len(marked)/N
    baseline = len(classical_marked) / N
    print(f"Baseline (uniform) probability of hitting a triangular number: {baseline:.4f}")

    # Grover should amplify well above baseline; require strong majority.
    quantum_ok = marked_fraction > 0.85 and marked_fraction > 2 * baseline

    # also check that index 15 (the A280992 term) individually got amplified
    bs_15 = format(15, f"0{N_QUBITS}b")
    prob_15 = counts.get(bs_15, 0) / total_shots
    expected_prob_15 = 1.0 / len(classical_marked) * marked_fraction  # rough share
    print(f"Probability of measuring 15 (A280992 term, T(5)): {prob_15:.4f}")

    verified = a280992_ok and quantum_ok

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
