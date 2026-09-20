"""
Erdos problem #587 -- quantum-testable instance.

OEIS sequence used: A372040
  "Smallest k such that there is an n-element subset of {1, 2, ..., k}
   that does not contain a (nonempty) subset that sums to a square."
  Known terms include a(1)=2, a(2)=3, a(3)=5, a(4)=8, a(5)=12, ...

Classical property tested (computed from first principles in this script,
not copied from OEIS):

  For n = 3, verify a(3) = 5 by exhaustive classical search:
    (a) among all 3-element subsets of {1,2,3,4} (k=4), NONE is
        "square-sum-free" (every one has some nonempty sub-subset whose
        elements sum to a perfect square) -- so k=4 fails.
    (b) among all 3-element subsets of {1,2,3,4,5} (k=5), AT LEAST ONE
        is square-sum-free -- so k=5 succeeds.
  Together (a)+(b) establish a(3) = 5, matching OEIS A372040.

Quantum circuit: Grover's search over the 5-qubit space of subsets of
{1,2,3,4,5} (qubit i set means element i+1 is included). The oracle,
built from the classical brute-force computation above, phase-flips
exactly the "good" bitstrings: 3-element subsets of {1..5} that are
square-sum-free. Grover amplitude amplification is run with the optimal
number of iterations for the known number of marked states, and the
ideal AerSimulator is sampled. The most frequent measured bitstring is
decoded back to a subset and checked classically against the same
square-sum-free property used to build the oracle (and against the
independently precomputed set of good subsets) -- this is the PASS/FAIL
comparison between the quantum result and the classical answer.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS values copied in).
# ---------------------------------------------------------------------------

def is_perfect_square(m: int) -> bool:
    if m < 0:
        return False
    r = math.isqrt(m)
    return r * r == m


def square_sum_free(subset: tuple) -> bool:
    """True iff no nonempty sub-subset of `subset` sums to a perfect square."""
    n = len(subset)
    for r in range(1, n + 1):
        for sub in combinations(subset, r):
            if is_perfect_square(sum(sub)):
                return False
    return True


def good_subsets(k: int, n: int):
    """All n-element subsets of {1,...,k} that are square-sum-free."""
    return [c for c in combinations(range(1, k + 1), n) if square_sum_free(c)]


N_ELEMENTS = 3  # subset size n

# Step (a): k = 4 must have NO good subset.
good_k4 = good_subsets(4, N_ELEMENTS)
assert good_k4 == [], f"expected no square-sum-free 3-subset of {{1..4}}, found {good_k4}"

# Step (b): k = 5 must have AT LEAST ONE good subset.
good_k5 = good_subsets(5, N_ELEMENTS)
assert len(good_k5) >= 1, "expected at least one square-sum-free 3-subset of {1..5}"

print("Classical check: good 3-subsets of {1..4}:", good_k4)
print("Classical check: good 3-subsets of {1..5}:", good_k5)
print("=> a(3) = 5, consistent with OEIS A372040.")

CLASSICAL_ANSWER = 5  # a(3), derived above, not copied

# ---------------------------------------------------------------------------
# 2. Build the Grover search instance over 5 qubits (subsets of {1..5}).
# ---------------------------------------------------------------------------

K = 5           # working with {1,...,K}
NUM_QUBITS = K  # one qubit per element; bit i (0-indexed) <-> element i+1

marked_bitstrings = []
for bits in range(2 ** NUM_QUBITS):
    if bin(bits).count("1") != N_ELEMENTS:
        continue
    subset = tuple(i + 1 for i in range(NUM_QUBITS) if (bits >> i) & 1)
    if square_sum_free(subset):
        marked_bitstrings.append(bits)

# Cross-check against the independently computed good_k5 list.
marked_subsets = sorted(
    tuple(i + 1 for i in range(NUM_QUBITS) if (bits >> i) & 1)
    for bits in marked_bitstrings
)
assert marked_subsets == sorted(good_k5), "oracle marking disagrees with classical search"

M = len(marked_bitstrings)
print(f"Marked (square-sum-free) bitstrings among {2**NUM_QUBITS}: {M} -> {marked_subsets}")


def bits_to_binstr(bits: int, num_qubits: int) -> str:
    return format(bits, f"0{num_qubits}b")


def apply_multi_controlled_z_on_bitstring(qc: QuantumCircuit, bits: int, num_qubits: int):
    """Flip the phase of the single computational basis state `bits`."""
    binstr = bits_to_binstr(bits, num_qubits)  # MSB = qubit num_qubits-1
    # X on qubits that should be 0, so the target pattern becomes all-ones.
    zero_qubits = [q for q in range(num_qubits) if binstr[num_qubits - 1 - q] == "0"]
    for q in zero_qubits:
        qc.x(q)
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    for q in zero_qubits:
        qc.x(q)


def oracle(qc: QuantumCircuit, num_qubits: int):
    for bits in marked_bitstrings:
        apply_multi_controlled_z_on_bitstring(qc, bits, num_qubits)


def diffuser(qc: QuantumCircuit, num_qubits: int):
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))


# Optimal number of Grover iterations for M marked states out of 2^n.
N_STATES = 2 ** NUM_QUBITS
iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M)))
print(f"Grover iterations: {iterations}")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    oracle(qc, NUM_QUBITS)
    diffuser(qc, NUM_QUBITS)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: rightmost char of the count key is qubit 0.
top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])
top_bits = int(top_bitstring, 2)
top_subset = tuple(i + 1 for i in range(NUM_QUBITS) if (top_bits >> i) & 1)

print(f"Top measured outcome: {top_bitstring} (count {top_count}/{shots}) -> subset {top_subset}")

# Fraction of shots landing on ANY marked bitstring (amplification check).
marked_shots = sum(c for b, c in counts.items() if int(b, 2) in marked_bitstrings)
amplified_fraction = marked_shots / shots
print(f"Fraction of shots on marked states: {amplified_fraction:.3f} "
      f"(uniform baseline would be {M / N_STATES:.3f})")

# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

quantum_found_valid_subset = (
    len(top_subset) == N_ELEMENTS
    and set(top_subset).issubset(set(range(1, K + 1)))
    and square_sum_free(top_subset)
    and top_subset in good_k5
)
quantum_amplified = amplified_fraction > (M / N_STATES) * 1.5  # clearly above uniform

verified_against_classical = quantum_found_valid_subset and quantum_amplified

print(f"quantum_found_valid_subset = {quantum_found_valid_subset}")
print(f"quantum_amplified = {quantum_amplified}")
print(f"classical a(3) = {CLASSICAL_ANSWER} (OEIS A372040)")

if verified_against_classical:
    print("PASS")
else:
    print("FAIL")
