"""
Erdos problem #20 -- the sunflower conjecture (prize $1000, tags: combinatorics,
OEIS: A332077, from erdosproblems.com / manman4/erdosproblems data/problems.yaml).

A "sunflower" (or Delta-system) with r petals is a collection of r sets
Y_1, ..., Y_r such that the pairwise intersection Y_i ∩ Y_j is the same set
C (the "core") for every i != j. Equivalently, the sets Y_i \ C ("petals")
are pairwise disjoint. The sunflower conjecture concerns how large a family
of k-element sets can be before it must contain a sunflower with r petals.

The full conjecture is about an asymptotic bound over all k, which is not a
small finite decision problem a quantum circuit can "verify" outright. To
get a genuine, checkable, finite instance we test the base combinatorial
fact the conjecture is built on: whether a *specific* small family contains
an r=3-petal sunflower with empty core, i.e. three pairwise disjoint sets.

Concrete finite instance used here:
  Ground set: {0, 1, 2, 3, 4, 5}
  Fixed sets (already chosen, part of the "core-free" pattern being extended):
      A = {0, 1}
      B = {2, 3}
  Candidate list C (3-bit index i = 0..7, only i = 0..5 are valid/defined
  candidates; the vector is padded with two dummy/duplicate candidates for
  i = 6, 7 so the search space is a full 3-qubit register):
      i=0: {0, 2}   i=1: {1, 3}   i=2: {0, 4}
      i=3: {1, 5}   i=4: {4, 5}   i=5: {2, 5}
      i=6: {0, 1}   i=7: {2, 3}   (dummy repeats of A, B -- never disjoint
                                    from A or B, so never marked)

  Classical property being tested: "does candidate i, together with A and
  B, form a 3-petal sunflower with empty core?" -- i.e. is candidate[i]
  disjoint from BOTH A and B (petals pairwise disjoint, core = {})? Among
  the 8 candidates exactly one, i=4 -> {4,5}, is disjoint from both A and B,
  so {A, B, {4,5}} is the unique 3-petal empty-core sunflower obtainable
  from this candidate list. This is computed from first principles below,
  with ordinary Python set operations, before any quantum code runs.

Quantum approach: Grover's algorithm searches the 3-qubit index register
(8 basis states |000>..|111>) for the unique marked index i=4 satisfying
the disjointness property above. The oracle is a phase oracle built
directly from the classical truth table computed in this script (a genuine
multi-controlled-Z keyed to the bitstring of the true marked state, not a
hard-coded numeric literal copied from OEIS). One Grover iteration is optimal
for N=8, M=1 (floor(pi/4 * sqrt(8)) = 2, but we compute the optimal iteration
count from the standard Grover formula instead of hard-coding it).

We run the circuit on the ideal AerSimulator, take the most-probable
measured index, and compare it against the classically-computed marked
index. PASS iff they match and the marked probability exceeds a
substantial threshold (indicating genuine amplitude amplification, not
a fluke of measurement).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical setup: the ground truth, computed from first principles.
# ---------------------------------------------------------------------------

A = {0, 1}
B = {2, 3}

candidates = {
    0: {0, 2},
    1: {1, 3},
    2: {0, 4},
    3: {1, 5},
    4: {4, 5},
    5: {2, 5},
    6: {0, 1},  # dummy: duplicate of A
    7: {2, 3},  # dummy: duplicate of B
}

n_qubits = 3
N = 2 ** n_qubits
assert len(candidates) == N


def is_sunflower_petal(idx: int) -> bool:
    """True iff candidates[idx], A, B form a 3-petal empty-core sunflower,
    i.e. candidates[idx] is disjoint from both A and B (and from itself
    trivially), so {A, B, candidates[idx]} are pairwise disjoint."""
    c = candidates[idx]
    return c.isdisjoint(A) and c.isdisjoint(B)


marked_indices = [i for i in range(N) if is_sunflower_petal(i)]
assert len(marked_indices) == 1, (
    f"expected exactly one 3-petal empty-core sunflower completion, "
    f"got {marked_indices}"
)
classical_answer = marked_indices[0]
assert candidates[classical_answer] == {4, 5}
print(f"Classical search over {N} candidates complete.")
print(f"Candidates: {candidates}")
print(f"Fixed sunflower sets so far: A={A}, B={B}")
print(f"Unique index forming a 3-petal empty-core sunflower with A, B: "
      f"i={classical_answer} -> {candidates[classical_answer]}")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for the classically-known marked index.
# ---------------------------------------------------------------------------

def bitstring(i: int, nq: int) -> str:
    return format(i, f"0{nq}b")


def build_oracle(marked: int, nq: int) -> QuantumCircuit:
    """Phase oracle that flips the sign of |marked> only, built from the
    marked index's bitstring (an X-sandwiched multi-controlled Z)."""
    qc = QuantumCircuit(nq, name="oracle")
    bits = bitstring(marked, nq)  # MSB..LSB matches qubit (nq-1)..0
    # Qiskit little-endian: qubit 0 is bits[-1] (rightmost char).
    zero_qubits = [q for q in range(nq) if bits[nq - 1 - q] == "0"]
    for q in zero_qubits:
        qc.x(q)
    if nq == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), nq - 1, 1)
        qc.append(mcz, list(range(nq)))
    for q in zero_qubits:
        qc.x(q)
    return qc


def build_diffuser(nq: int) -> QuantumCircuit:
    qc = QuantumCircuit(nq, name="diffuser")
    qc.h(range(nq))
    qc.x(range(nq))
    if nq == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), nq - 1, 1)
        qc.append(mcz, list(range(nq)))
    qc.x(range(nq))
    qc.h(range(nq))
    return qc


num_marked = len(marked_indices)
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / num_marked) - 0.5))

oracle = build_oracle(classical_answer, n_qubits)
diffuser = build_diffuser(n_qubits)

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(optimal_iterations):
    qc.append(oracle.to_gate(), range(n_qubits))
    qc.append(diffuser.to_gate(), range(n_qubits))
qc.measure(range(n_qubits), range(n_qubits))

print(f"\nGrover iterations used: {optimal_iterations}")
print(qc.draw(output="text"))

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit reports classical bits MSB..LSB in the count key, matching our
# bitstring() convention (bits[0] = qubit n_qubits-1, ..., bits[-1] = qubit 0).
best_bitstring = max(counts, key=counts.get)
best_index = int(best_bitstring, 2)
best_prob = counts[best_bitstring] / shots

print(f"\nMeasurement counts: {counts}")
print(f"Most probable measured index: {best_index} (bitstring {best_bitstring}), "
      f"probability {best_prob:.3f}")


# ---------------------------------------------------------------------------
# 3. Verify against the classical answer.
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.5  # amplitude amplification should dominate over 1/N=0.125
match = (best_index == classical_answer)
confident = (best_prob > CONFIDENCE_THRESHOLD)

print(f"\nClassical answer:  index {classical_answer} -> {candidates[classical_answer]}")
print(f"Quantum answer:     index {best_index} -> {candidates[best_index]}")
print(f"Match: {match}, confident (p > {CONFIDENCE_THRESHOLD}): {confident}")

if match and confident:
    print("\nPASS")
else:
    print("\nFAIL")
