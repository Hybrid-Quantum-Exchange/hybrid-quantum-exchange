"""
Erdos problem #734 -- quantum-testable-sequence attempt.

LIMITATION (read first): problem #734's entry in erdosproblems/data/problems.yaml
lists oeis: ["possible"] -- there is no actual OEIS sequence id attached to this
problem (the string "possible" is a metadata placeholder, not an id), and the
only tag is "combinatorics" with no further formal statement available in the
read-only clone. There is therefore no real, specific integer sequence to
derive a genuine finite computable property from for problem 734 itself. Per
instructions, this script is a best-honest-effort fallback rather than a
fabricated match: it builds a REAL, correct Grover-search quantum circuit for
a small, well-defined, genuinely-computable combinatorial search problem (in
the same spirit as problem 734's "combinatorics" tag -- subset-sum /
subset-selection search, a canonical combinatorics decision problem), and
verifies the quantum result against a brute-force classical computation done
from first principles in this script.

Chosen finite instance (NOT derived from a specific OEIS sequence, because
none exists for #734): given the 4-element set S = {3, 5, 7, 9} and target
T = 12, find which 4-bit subset-indicator strings x in {0,1}^4 select a
subset of S summing exactly to T. This is decided classically first, then a
Grover oracle marking exactly those bitstrings is built and run on
AerSimulator; the most frequently measured bitstring(s) must match the
classical solution set for the script to print PASS.

Classical answer for S = {3,5,7,9}, T = 12 (computed by brute force below):
  {3,9} -> bits (x0..x3 select 3,5,7,9) = 1,0,0,1 -> "1001" (and its
  reverse/bit-order twin depending on Qiskit's little-endian convention,
  handled explicitly in the classical check below)
  {5,7} -> "0110"
So there are exactly two solutions among the 16 possible subsets.

Reported honestly: ran_ok reflects whether the script executed and printed
PASS; verified_against_classical reflects whether the quantum measurement
distribution's top outcomes exactly equal the classically brute-forced
solution set. Because the *instance* is a stand-in (no real OEIS sequence
exists for #734), this is flagged as best-effort, not a genuine test of
problem 734's own mathematical content.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_subset_sum_solutions(values, target):
    """Brute-force every subset of `values`; return the set of 4-bit
    strings (bit i = 1 iff values[i] is included) whose subset sums to
    `target`. Computed from first principles (plain enumeration)."""
    n = len(values)
    solutions = set()
    for bits in itertools.product([0, 1], repeat=n):
        s = sum(v for v, b in zip(values, bits) if b)
        if s == target:
            # bits[0] is the least-significant conceptual bit (values[0]);
            # store as a string with index 0 first for clarity.
            solutions.add("".join(str(b) for b in bits))
    return solutions


def build_oracle(n_qubits, solutions):
    """Phase-flip oracle marking each classical solution bitstring."""
    qc = QuantumCircuit(n_qubits)
    for sol in solutions:
        # sol[i] corresponds to qubit i (values[i]); Qiskit circuit qubit
        # order i = same index. Multi-controlled Z on the |sol> state:
        # flip qubits that are 0 in sol, apply MCZ, flip back.
        zero_positions = [i for i, b in enumerate(sol) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    values = [3, 5, 7, 9]
    target = 12
    n = len(values)

    solutions = classical_subset_sum_solutions(values, target)
    print(f"Classical brute-force solutions (subset-sum = {target}): {sorted(solutions)}")
    assert solutions == {"1001", "0110"}, "classical computation did not match expectation"

    N = 2 ** n
    M = len(solutions)
    # Optimal number of Grover iterations for N items, M marked.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"N={N}, M={M}, Grover iterations={iterations}")

    oracle = build_oracle(n, solutions)
    diffuser = build_diffuser(n)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n), range(n))

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical register bit order in count keys is reversed
    # relative to qubit index (qubit 0 is the rightmost character).
    # Normalize counts keys back to our qubit-index-first convention.
    normalized_counts = {}
    for key, c in counts.items():
        qubit_order_key = key[::-1]  # now index 0 first, matching `values`/`solutions`
        normalized_counts[qubit_order_key] = normalized_counts.get(qubit_order_key, 0) + c

    sorted_outcomes = sorted(normalized_counts.items(), key=lambda kv: -kv[1])
    print("Top measured outcomes (qubit-index-first order):", sorted_outcomes[:6])

    top_m = set(k for k, _ in sorted_outcomes[:M])
    total_top_prob = sum(c for k, c in normalized_counts.items() if k in solutions) / shots

    print(f"Total probability mass on classical solutions: {total_top_prob:.3f}")
    verified = (top_m == solutions) and (total_top_prob > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
