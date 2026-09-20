"""
Erdos problem #289 (unit fractions / number theory; erdosproblems.com #289).

Source metadata (data/problems.yaml, manman4/erdosproblems, as of 2026-09-19):
    number: "289"
    tags: ["number theory", "unit fractions"]
    oeis: ["N/A"]

LIMITATION: problem #289 carries no OEIS sequence id in the source data
(oeis: ["N/A"]), so there is no OEIS-derived sequence to test membership in.
In its place, this script tests a small, finite, genuinely computable
property in the same subject area as the problem's own tags (unit
fractions / Egyptian-fraction representations of 1), which is the closest
honest substitute available without an OEIS id to anchor to:

    PROPERTY TESTED: among the distinct unit fractions with denominators
    drawn from D = {2, 3, 4, 5, 6}, does there exist a subset that sums
    exactly to 1?  (I.e. is 1 expressible as a sum of distinct unit
    fractions 1/d, d in D?)

CLASSICAL ANSWER (computed here from first principles, not copied from
anywhere): brute-force enumeration of all 2^5 = 32 subsets of D, scaling
to the common denominator L = lcm(2,3,4,5,6) = 60 and working in exact
integers t_i = L / d_i = [30, 20, 15, 12, 10], finds exactly one subset
summing to L = 60: {2, 3, 6}  (1/2 + 1/3 + 1/6 = 1). No other subset of D
works. So the unique marked bitstring, with qubit i <-> element D[i]
selected, is bits (q0, q1, q2, q3, q4) = (1, 1, 0, 0, 1) i.e. elements
{2, 3, 6} chosen, {4, 5} not.

QUANTUM CIRCUIT: Grover's algorithm on 5 qubits searches the 32-element
subset space for that unique marked bitstring. The oracle is built as a
phase-flip (multi-controlled Z, with X-gates on the qubits that must be 0)
on exactly the bitstring found by the classical brute force above -- the
oracle is derived from, not fabricated independently of, that classical
computation. The optimal number of Grover iterations for N=32, M=1 marked
state is round(pi/4 * sqrt(32)) = 4. The circuit is run on the ideal
AerSimulator and the most frequently measured bitstring is compared to
the classical answer.

PASS means: the bitstring Grover returns as most likely, decoded back to
a subset of D, sums its unit fractions to exactly 1 (verified again in
exact Fraction arithmetic), and matches the unique classical solution.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import gcd, pi
from functools import reduce

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, brute force over all subsets)
# ---------------------------------------------------------------------------

D = [2, 3, 4, 5, 6]
N = len(D)


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


L = reduce(lcm, D)  # common denominator = 60
T = [L // d for d in D]  # integer-scaled unit fractions = [30, 20, 15, 12, 10]

classical_solutions = []
for r in range(1, N + 1):
    for combo in combinations(range(N), r):
        if sum(T[i] for i in combo) == L:
            classical_solutions.append(combo)

assert len(classical_solutions) == 1, (
    f"expected a unique subset summing unit fractions to 1, found {classical_solutions}"
)
solution_indices = classical_solutions[0]  # (0, 1, 4) -> denominators {2, 3, 6}

# Double-check in exact rational arithmetic (independent of the integer scaling above).
frac_sum = sum(Fraction(1, D[i]) for i in solution_indices)
assert frac_sum == Fraction(1, 1), f"unit fractions for {solution_indices} sum to {frac_sum}, not 1"

# Marked bitstring: bit i (from qubit i, little-endian) is 1 iff index i is in the solution.
marked_bits = ["1" if i in solution_indices else "0" for i in range(N)]
marked_bitstring_little_endian = "".join(marked_bits)  # q0 q1 q2 q3 q4
# Qiskit prints classical register bits big-endian (q{N-1}...q0), so build that form too.
marked_bitstring_qiskit_order = marked_bitstring_little_endian[::-1]

print("Classical brute force over all subsets of D =", D)
print("  scaled terms T = L/d =", T, " (L =", L, ")")
print("  unique solution: indices", solution_indices,
      "-> denominators", [D[i] for i in solution_indices],
      "-> 1/2+1/3+1/6+... =", frac_sum)
print("  marked bitstring (q0..q4, little-endian):", marked_bitstring_little_endian)
print("  marked bitstring (Qiskit print order, q4..q0):", marked_bitstring_qiskit_order)


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover's algorithm searching for that one bitstring
# ---------------------------------------------------------------------------

def build_oracle(n: int, marked: str) -> QuantumCircuit:
    """Phase-flip oracle marking the single bitstring `marked` (little-endian,
    marked[i] is the value of qubit i)."""
    qc = QuantumCircuit(n, name="Oracle")
    zero_qubits = [i for i, b in enumerate(marked) if b == "0"]
    for q in zero_qubits:
        qc.x(q)
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
        qc.h(n - 1)
    for q in zero_qubits:
        qc.x(q)
    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


num_iterations = round((pi / 4) * (2 ** N) ** 0.5)  # ~4 for N=5 qubits, 1 marked state
print(f"\nGrover iterations used: {num_iterations} (32 states, 1 marked)")

oracle = build_oracle(N, marked_bitstring_little_endian)
diffuser = build_diffuser(N)

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(num_iterations):
    qc.append(oracle.to_instruction(), range(N))
    qc.append(diffuser.to_instruction(), range(N))
qc.measure(range(N), range(N))

sim = AerSimulator()
shots = 4096
qc = qc.decompose().decompose()
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])
print(f"\nTop measured bitstring (Qiskit order, q{N-1}..q0): {top_bitstring}  "
      f"({top_count}/{shots} shots = {top_count/shots:.1%})")

# Convert Qiskit's printed (big-endian) bitstring back to little-endian q0..q_{N-1}.
top_little_endian = top_bitstring[::-1]
top_indices = tuple(i for i, b in enumerate(top_little_endian) if b == "1")
top_denominators = sorted(D[i] for i in top_indices)
top_sum = sum(Fraction(1, d) for d in top_denominators)

print(f"  decodes to denominators {top_denominators}, unit-fraction sum = {top_sum}")


# ---------------------------------------------------------------------------
# 3. Verify quantum result against the classical answer
# ---------------------------------------------------------------------------

quantum_matches_classical = (
    top_indices == solution_indices
    and top_sum == Fraction(1, 1)
    and top_count / shots > 0.5  # Grover should concentrate probability on the marked state
)

print("\nExpected (classical) denominators:", [D[i] for i in solution_indices])
print("Quantum-found denominators:       ", top_denominators)
print("Measurement concentration on marked state:", f"{top_count/shots:.1%}")

if quantum_matches_classical:
    print("\nPASS: Grover search found the unique unit-fraction subset summing to 1, "
          "matching the classical brute-force answer.")
else:
    print("\nFAIL: quantum result did not match the classical answer.")

assert quantum_matches_classical, "quantum result does not match classical answer"
