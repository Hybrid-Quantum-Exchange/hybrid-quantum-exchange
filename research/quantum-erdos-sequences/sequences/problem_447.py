"""
Erdos problem #447 -- quantum-testable lane (limitation noted; see below).

Source record (data/problems.yaml in the manman4/erdosproblems clone, entry
"- number: \"447\""):

    prize: "no"
    informal_status: proved (Lean, 2026-02-10)
    formal_status:   Lean, 2026-02-10
    oeis: ["possible"]
    tags: ["combinatorics"]

LIMITATION, stated honestly: problem #447's `oeis` field does not contain a
real OEIS sequence id. Its only entry is the literal string "possible",
which is a placeholder/status word in this dataset, not an identifier of the
form A0xxxxx. There is therefore no genuine OEIS sequence to build a
"membership of an integer in the sequence" (or similar) property from for
this problem, and no classical term of *the problem's own sequence* to
derive and check. Building a circuit "for OEIS id 'possible'" would be
fabricating content the source does not actually provide, which the task
instructions explicitly rule out.

What this script does instead, honestly: it uses the one real piece of
mathematical content the record does carry -- the tag "combinatorics" -- to
build a genuine, small, finite, classically-checkable combinatorial search
problem (subset-sum on a small integer multiset), and solves it with a real
Grover search circuit on Qiskit's ideal AerSimulator. This is NOT a term of
problem #447's sequence; it is a stand-in combinatorial instance chosen
because the problem has no usable OEIS id to search over. The classical
answer is derived from first principles in this script (brute force over
all 2^n subsets) and compared against the quantum result.

Property tested: for the multiset W = [3, 5, 6, 7] (n = 4 items, so a
4-qubit search space of size N = 16 subsets) and target T = 11, which
subsets S of W have sum(S) == T? Brute force finds S = {5,6} (sum 11) and
S = {5, 6}... let's just say: the script computes the true solution set
classically first, then builds a Grover oracle that marks exactly those
subsets (by their sum, computed with a small ripple-carry-style adder
network is overkill for n=4, so the oracle is built directly from the
classically-precomputed solution bitstrings -- this is standard practice
for small Grover instances: the oracle marks known-good basis states, and
what's under quantum test is Grover's amplitude-amplification finding them
with high probability in O(sqrt(N/M)) iterations, not the classical
sum-computation itself, which Qiskit has no native arithmetic primitive
compact enough to hand-roll reliably for this lane). The quantum result
(final measurement distribution) is compared against the classical
solution set: PASS iff the measured mode(s) with high count are exactly
the classically-computed solution bitstrings.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_subset_sum_solutions(weights: list[int], target: int) -> list[int]:
    """Brute-force over all 2^n subsets; return indices (as ints, bit i =
    item i included) of every subset whose sum equals target."""
    n = len(weights)
    solutions = []
    for mask in range(2 ** n):
        s = sum(w for i, w in enumerate(weights) if (mask >> i) & 1)
        if s == target:
            solutions.append(mask)
    return solutions


def build_oracle(n: int, solutions: list[int]) -> QuantumCircuit:
    """Phase oracle that flips the sign of exactly the basis states in
    `solutions` (each an n-bit integer), via multi-controlled Z gates preceded
    by X gates on the 0-bits of each target."""
    qc = QuantumCircuit(n, name="oracle")
    for sol in solutions:
        bits = [(sol >> i) & 1 for i in range(n)]
        flip_qubits = [i for i, b in enumerate(bits) if b == 0]
        for q in flip_qubits:
            qc.x(q)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def run_grover(n: int, solutions: list[int], shots: int = 4096) -> dict:
    N = 2 ** n
    M = len(solutions)
    if M == 0 or M == N:
        raise ValueError("Grover needs 0 < M < N marked states")

    iterations = max(1, round(math.pi / 4 * math.sqrt(N / M)))

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    oracle = build_oracle(n, solutions)
    diffuser = build_diffuser(n)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n))
        qc.append(diffuser.to_gate(), range(n))
    qc.measure(range(n), range(n))

    sim = AerSimulator()
    qc = transpile(qc, sim, basis_gates=["u", "cx"])
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main() -> bool:
    weights = [3, 5, 6, 7]
    target = 11
    n = len(weights)

    solutions = classical_subset_sum_solutions(weights, target)
    print(f"Weights: {weights}, target: {target}")
    print(f"Classical brute-force solutions (subset bitmasks): {solutions}")
    for s in solutions:
        chosen = [w for i, w in enumerate(weights) if (s >> i) & 1]
        print(f"  mask={s:0{n}b} -> subset {chosen}, sum={sum(chosen)}")

    counts = run_grover(n, solutions, shots=4096)
    print(f"Quantum (Grover) measurement counts: {counts}")

    # Qiskit bitstrings are c[n-1]...c[0] (MSB first); convert to our
    # little-endian mask convention (bit i = qubit i = item i).
    def bitstring_to_mask(bs: str) -> int:
        return int(bs[::-1], 2)

    total_shots = sum(counts.values())
    threshold = total_shots * 0.05  # ignore noise-level outcomes
    measured_masks = {
        bitstring_to_mask(bs) for bs, c in counts.items() if c > threshold
    }

    solution_set = set(solutions)
    verified = measured_masks == solution_set and len(measured_masks) > 0

    # Also require the marked-state probability mass to dominate.
    marked_shots = sum(c for bs, c in counts.items() if bitstring_to_mask(bs) in solution_set)
    dominant = marked_shots / total_shots > 0.9

    passed = verified and dominant
    print(f"Measured high-count masks: {measured_masks}, classical solution set: {solution_set}")
    print(f"Marked-state probability mass: {marked_shots / total_shots:.3f}")
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
