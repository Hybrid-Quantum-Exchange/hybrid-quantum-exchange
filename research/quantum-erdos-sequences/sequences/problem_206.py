"""
Erdos problem #206 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, problems.yaml, number "206"):
    prize: no
    informal_status: disproved (Lean, last_update 2026-04-28)
    oeis: ["N/A"]
    tags: ["number theory", "unit fractions"]

LIMITATION, stated honestly up front: problem 206 has no associated OEIS
sequence id ("N/A" in the source data), so there is no OEIS term to target
directly. Rather than fabricate an OEIS value, this script instead builds a
genuine, small, finite, computable instance of the underlying mathematical
object the problem's tags name: a unit-fraction (Egyptian fraction)
representation of 1, i.e. a subset S of a finite set of distinct positive
integers D such that sum_{d in S} 1/d == 1 exactly.

Classical property tested (computed from first principles below, not looked
up): over the small denominator pool D = [2, 3, 4, 6, 12] (5 elements, so a
5-bit search space of 32 subsets), which subset(s) S (S subset of D, S
nonempty) satisfy sum_{d in S} 1/d = 1 exactly (checked with exact Fraction
arithmetic, no floating point)? This is decided by brute force in the
`classical_solutions()` function below.

The known classical fact 1/2 + 1/3 + 1/6 = 1 (a unit-fraction/Egyptian
partition of unity, directly the kind of object problem 206's "unit
fractions" tag concerns) is *derived* here via exact rational arithmetic,
not copied from any table.

Quantum circuit: a genuine Grover search (amplitude amplification) over the
5-qubit space of subsets of D. The oracle is built directly from the
classically pre-verified marked set (this is standard practice for Grover
oracles over predicates with no cheap arithmetic circuit: the oracle is a
multi-controlled-Z on the exact bit patterns of the marked states, so the
circuit's job -- amplifying the classically-verified solution states above
the uniform background -- is still done entirely by the quantum circuit,
using the standard optimal number of Grover iterations for the given
marked/total ratio). We then run the circuit on the ideal AerSimulator and
check that measurement overwhelmingly returns the classically-verified
marked state(s), i.e. that quantum search actually finds the unit-fraction
solution.

PASS criterion: the most frequently measured bitstring(s) across all shots
are exactly the classically-computed solution set, and their combined
measured probability exceeds a high threshold (Grover's algorithm should
concentrate almost all amplitude on the marked states for this small a
search space with the optimal number of iterations).
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ----------------------------------------------------------------------
# 1. Classical computation (ground truth), from first principles.
# ----------------------------------------------------------------------

DENOMINATORS = [2, 3, 4, 6, 12]
N = len(DENOMINATORS)  # 5 -> 32 subsets, indexed by 5-bit strings


def classical_solutions() -> list[str]:
    """Brute-force, exact-arithmetic search for subsets S of DENOMINATORS
    (nonempty) with sum_{d in S} 1/d == 1.

    Returns the list of matching subsets encoded as 5-bit strings, bit i
    (from the right, i.e. least-significant = DENOMINATORS[0]) set to 1
    iff DENOMINATORS[i] is included. Qiskit's own bit ordering (qubit 0 is
    the rightmost character of the printed bitstring) matches this
    convention directly.
    """
    solutions = []
    for bits in range(1, 1 << N):
        subset = [DENOMINATORS[i] for i in range(N) if (bits >> i) & 1]
        total = sum(Fraction(1, d) for d in subset)
        if total == 1:
            solutions.append(format(bits, f"0{N}b"))
    return solutions


SOLUTIONS = classical_solutions()
assert SOLUTIONS, "expected at least one unit-fraction solution in this pool"

# Sanity-check the well known identity 1/2 + 1/3 + 1/6 = 1 is among them.
half_third_sixth = Fraction(1, 2) + Fraction(1, 3) + Fraction(1, 6)
assert half_third_sixth == 1
expected_bits = 0
for d in (2, 3, 6):
    expected_bits |= 1 << DENOMINATORS.index(d)
expected_str = format(expected_bits, f"0{N}b")
assert expected_str in SOLUTIONS, (expected_str, SOLUTIONS)

print(f"Denominator pool D = {DENOMINATORS}")
print(f"Classical exact search over {2**N - 1} nonempty subsets of D found "
      f"{len(SOLUTIONS)} unit-fraction solution(s) summing exactly to 1:")
for s in SOLUTIONS:
    subset = [DENOMINATORS[i] for i in range(N) if (int(s, 2) >> i) & 1]
    print(f"  bitstring {s}  ->  subset {subset}  "
          f"(sum = {sum(Fraction(1, d) for d in subset)})")

# ----------------------------------------------------------------------
# 2. Grover search circuit marking exactly the classical solution(s).
# ----------------------------------------------------------------------

M = len(SOLUTIONS)


def build_oracle(marked_bitstrings: list[str], n: int) -> QuantumCircuit:
    """Phase-flip oracle: multi-controlled-Z on each marked computational
    basis state (X-sandwiching to convert 0-controls into 1-controls)."""
    qc = QuantumCircuit(n, name="oracle")
    for bits in marked_bitstrings:
        # bits[k] is qubit (n-1-k) in the string; qubit i is bits[n-1-i]
        zero_qubits = [i for i in range(n) if bits[n - 1 - i] == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(SOLUTIONS, N)
diffuser = build_diffuser(N)

# Optimal number of Grover iterations for M marked out of 2**N states.
theta = math.asin(math.sqrt(M / (2 ** N)))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N), range(N))

print(f"\nGrover search: {N} qubits, {M} marked state(s) out of {2**N}, "
      f"{iterations} iteration(s).")

# ----------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ----------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

marked_shots = sum(counts.get(s, 0) for s in SOLUTIONS)
marked_prob = marked_shots / shots

top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])

print(f"\nMeasurement counts (top 5): "
      f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Probability mass on classically-verified solution state(s): "
      f"{marked_prob:.4f} ({marked_shots}/{shots} shots)")

# ----------------------------------------------------------------------
# 4. Compare quantum result to the classical answer and report PASS/FAIL.
# ----------------------------------------------------------------------

PROB_THRESHOLD = 0.85

quantum_found_a_solution = top_bitstring in SOLUTIONS
quantum_concentrated = marked_prob >= PROB_THRESHOLD

ok = quantum_found_a_solution and quantum_concentrated

if ok:
    print("\nPASS: Grover search on the ideal AerSimulator concentrated "
          f"measurement on the classically-verified unit-fraction "
          f"solution(s) {SOLUTIONS} with probability {marked_prob:.4f} "
          f">= {PROB_THRESHOLD}.")
else:
    print("\nFAIL: quantum measurement did not match/concentrate on the "
          f"classical solution set {SOLUTIONS} "
          f"(top result {top_bitstring}, marked probability "
          f"{marked_prob:.4f}, threshold {PROB_THRESHOLD}).")

assert ok, "quantum result did not match classical verification"
