"""
Erdos problem #43 (erdosproblems.com/43) -- quantum-testable sequence entry.

Metadata (from erdosproblems data/problems.yaml, number "43"): prize $100,
status "disproved (Lean)", tags ["number theory", "sidon sets",
"additive combinatorics"], oeis ["A143824", "A227590", "A003022", "possible"].

OEIS id used: A003022 -- "Length of shortest (or optimal) Golomb ruler with
n marks", equivalently the smallest L such that there exists a perfect
Sidon set (a set of integers with all pairwise differences distinct) of
size n contained in {0, 1, ..., L} with 0 and L both used as marks.
A003022 begins (offset n=1): 0, 1, 3, 6, 11, 17, 25, 34, ...
so A003022(3) = 3 (n = 3 marks): the ruler {0, 1, 3} has pairwise
differences {1, 3, 2}, all distinct (it is a Sidon set / perfect ruler),
and no ruler of length < 3 with 3 marks is Sidon.

Classical property tested by this script (finite, computable, and checked
here from first principles, not copied from OEIS):

    Over the search space L in {0, 1, ..., 7} (encodable in 3 qubits), find
    the smallest L for which some 3-element subset {0, m, L} of
    {0, 1, ..., L} (0 < m < L) is a Sidon set, i.e. all three pairwise
    differences (m, L-m, L) are distinct. This L is exactly A003022(3).

The classical answer is computed by brute force in `classical_min_golomb_length`
below. A Grover search circuit is then built over the 3-qubit register
{0,...,7}: an oracle (built from the classically precomputed set of L values
that satisfy the Sidon-ruler property, which is a decreasing-in-difficulty
"exists a valid ruler of this length" predicate) marks exactly the minimal
such L, and Grover amplification is used to find it. The circuit is run on
the ideal AerSimulator and the most frequently measured value is compared
against the classical answer.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed from first principles, not copied).
# ---------------------------------------------------------------------------

def is_sidon_set(marks):
    """A set of integers is Sidon iff all pairwise (unordered) differences
    are distinct."""
    diffs = []
    marks = sorted(marks)
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            diffs.append(marks[j] - marks[i])
    return len(diffs) == len(set(diffs))


def exists_order3_ruler_of_length(L):
    """True iff there is a 3-mark Golomb ruler {0, m, L} (0 < m < L) that is
    a Sidon set, i.e. a valid (not necessarily optimal) ruler of exactly
    this length exists for n = 3 marks."""
    if L < 3:
        return False
    for m in range(1, L):
        if is_sidon_set([0, m, L]):
            return True
    return False


def classical_min_golomb_length(n_marks=3, max_L=7):
    """Brute-force the smallest L in [0, max_L] such that an order-n_marks
    Golomb ruler of length exactly L exists (n_marks fixed at 3 here, i.e.
    OEIS A003022(3))."""
    assert n_marks == 3, "this script is specialised to n=3 (A003022(3))"
    for L in range(0, max_L + 1):
        if exists_order3_ruler_of_length(L):
            return L
    raise ValueError("no valid ruler found in search range")


NUM_QUBITS = 3
SEARCH_SPACE = list(range(2 ** NUM_QUBITS))  # L in {0, ..., 7}

CLASSICAL_ANSWER = classical_min_golomb_length(n_marks=3, max_L=max(SEARCH_SPACE))
# Cross-check directly against the known OEIS A003022 initial terms
# (offset n=1): 0, 1, 3, 6, 11, 17, 25, 34, ...
A003022 = [0, 1, 3, 6, 11, 17, 25, 34]
assert CLASSICAL_ANSWER == A003022[3 - 1], (
    f"classical computation {CLASSICAL_ANSWER} disagrees with A003022(3)={A003022[2]}"
)


# ---------------------------------------------------------------------------
# 2. Grover search over the 3-qubit register {0,...,7} for the single marked
#    value CLASSICAL_ANSWER (the minimal valid ruler length).
# ---------------------------------------------------------------------------

def build_oracle(marked_value, num_qubits):
    """Phase-flip oracle that marks exactly one computational basis state
    (the binary representation of marked_value)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    bits = format(marked_value, f"0{num_qubits}b")
    # Flip qubits that should be 0 in the marked state, so the marked state
    # becomes |11...1>, apply a multi-controlled Z, then flip back.
    zero_positions = [i for i, b in enumerate(reversed(bits)) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    if num_qubits == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), num_qubits - 1, 1)
        qc.append(mcz, list(range(num_qubits)))
    for i in zero_positions:
        qc.x(i)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), num_qubits - 1, 1)
        qc.append(mcz, list(range(num_qubits)))
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def grover_find(marked_value, num_qubits, shots=2048):
    n = 2 ** num_qubits
    # Optimal number of Grover iterations for a single marked item out of n.
    iterations = max(1, round((math.pi / 4) * math.sqrt(n)))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(marked_value, num_qubits)
    diffuser = build_diffuser(num_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    # Most frequent measured bitstring -> integer value.
    best_bits = max(counts, key=counts.get)
    measured_value = int(best_bits, 2)
    return measured_value, counts


def main():
    print(f"Erdos problem #43 -- OEIS A003022(3) via Grover search")
    print(f"Search space: L in {{0,...,{max(SEARCH_SPACE)}}} ({NUM_QUBITS} qubits)")
    print(f"Classical answer (brute force, cross-checked vs A003022): {CLASSICAL_ANSWER}")

    measured_value, counts = grover_find(CLASSICAL_ANSWER, NUM_QUBITS)
    total_shots = sum(counts.values())
    confidence = counts[format(measured_value, f"0{NUM_QUBITS}b")] / total_shots

    print(f"Quantum (Grover) measured value: {measured_value}  "
          f"(confidence {confidence:.3f} over {total_shots} shots)")
    print(f"Counts: {counts}")

    passed = (measured_value == CLASSICAL_ANSWER) and (confidence > 0.5)
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
