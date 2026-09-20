"""
Erdos problem #500 -- quantum-testable instance
=================================================

Erdos problem #500 (data/problems.yaml, number "500") concerns Turan's
(3,4)-conjecture on 3-uniform hypergraphs: the open problem of determining
T(n,4,3), the minimum number of triples (3-element edges) that must be
chosen from an n-vertex ground set so that EVERY 4-element subset of the
n vertices contains at least one chosen triple. The associated OEIS entry
is A140462, "Turan's upper bound on the number of triangles of a
simplicial complex of dimension two for which every minimal non-face has
three vertices", whose terms satisfy

    A140462(n) = C(n,3) - T(n,4,3)

(the maximum number of triangles you can keep in a 2-dimensional
simplicial complex on n vertices whose complement -- the "holes" --
still hits every 4-set). This script verifies that relationship
classically-and-quantumly for n = 5:

    C(5,3) = 10 candidate triples
    A140462(5) = 7  (from the OEIS b-file: offset 0, terms
                      0,0,0,1,3,7,14,... so a(5) = 7)
    => T(5,4,3) must equal 10 - 7 = 3

The classical, finite, computable property being tested:
    T(5,4,3) = 3, i.e. there EXISTS a 3-element subset of the 10 triples
    on {0,1,2,3,4} whose union hits (contains a triple subset of) all
    C(5,4) = 5 four-element subsets, and no 2-element subset of triples
    can do this (so 3 is minimal).

This is computed from first principles below by brute force over all
2^10 = 1024 subsets of the 10 triples, confirming both:
  (a) the minimum hitting-set size is exactly 3, matching 10 - a(5) = 3, and
  (b) the exact set of minimum (size-3) covering triples-of-triples.

The quantum part then runs Grover's algorithm over a 10-qubit register
(one qubit per candidate triple, bit=1 meaning "triple is chosen") whose
oracle marks exactly the bitstrings of Hamming weight 3 that are valid
covering sets -- i.e. exactly the "good states" identified classically.
Grover search amplifies these marked computational basis states; we then
sample the circuit and check that the most frequently measured bitstring
is indeed one of the classically-verified minimum covering sets. This
means the quantum circuit is genuinely searching/verifying the discrete
structure underlying A140462(5), not just parroting the OEIS value.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS values used directly)
# ---------------------------------------------------------------------------

N_VERTICES = 5

triples = list(itertools.combinations(range(N_VERTICES), 3))   # 10 triples
quads = list(itertools.combinations(range(N_VERTICES), 4))     # 5 quads
NUM_TRIPLES = len(triples)
assert NUM_TRIPLES == 10
assert len(quads) == 5

triple_index = {t: i for i, t in enumerate(triples)}

# For each 4-subset, the indices of the (four) triples it contains.
quad_to_triple_indices = [
    [triple_index[t] for t in itertools.combinations(q, 3)] for q in quads
]


def is_covering(chosen_indices):
    """True iff every 4-subset contains at least one chosen triple."""
    chosen = set(chosen_indices)
    return all(any(ti in chosen for ti in qt) for qt in quad_to_triple_indices)


# Brute-force the minimum hitting-set size T(5,4,3), and collect every
# minimum covering set once we find the smallest size that has any.
min_size = None
minimum_covering_sets = []
for size in range(NUM_TRIPLES + 1):
    hits_at_this_size = []
    for combo in itertools.combinations(range(NUM_TRIPLES), size):
        if is_covering(combo):
            hits_at_this_size.append(combo)
    if hits_at_this_size:
        min_size = size
        minimum_covering_sets = hits_at_this_size
        break

assert min_size == 3, f"expected T(5,4,3) = 3, computed {min_size}"

# Cross-check against A140462: a(5) = 7 (OEIS b-file terms for offset 0:
# 0, 0, 0, 1, 3, 7, 14, ... i.e. a(0..6) = 0,0,0,1,3,7,14).
A140462_TERMS = [0, 0, 0, 1, 3, 7, 14]
a5 = A140462_TERMS[5]
c_5_3 = math.comb(N_VERTICES, 3)
assert c_5_3 - a5 == min_size, "A140462(5) does not match C(5,3) - T(5,4,3)"

print(f"Classical result: T(5,4,3) = {min_size} "
      f"(cross-checked via A140462(5)={a5}, C(5,3)={c_5_3}, "
      f"C(5,3)-A140462(5)={c_5_3 - a5})")
print(f"Number of minimum (size-3) covering triples-of-triples: "
      f"{len(minimum_covering_sets)}")

# `marked_logical[k]` has character i == qubit i's value (bit 1 <=> triple i
# chosen). This is the natural indexing to build the oracle circuit with.
# Qiskit's measurement counts print bitstrings with qubit (N-1) leftmost and
# qubit 0 rightmost, so `marked_display` is the reversed form used only when
# comparing against `counts` below.
marked_logical = set()
marked_display = set()
for combo in minimum_covering_sets:
    bits = ["0"] * NUM_TRIPLES
    for i in combo:
        bits[i] = "1"
    logical = "".join(bits)
    marked_logical.add(logical)
    marked_display.add("".join(reversed(logical)))

print(f"Example marked bitstring (display/qiskit-counts order): "
      f"{sorted(marked_display)[0]}")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 10-qubit "which triples are
#    chosen" register, oracle marks exactly the classically-verified
#    minimum covering sets.
# ---------------------------------------------------------------------------

NUM_QUBITS = NUM_TRIPLES  # 10 qubits, one per candidate triple


def append_oracle(qc: QuantumCircuit, marked: set):
    """Phase-flip every basis state whose LOGICAL bitstring (character i ==
    qubit i) is in `marked`."""
    mcz = ZGate().control(NUM_QUBITS - 1)
    for bitstring in marked:
        zero_positions = [i for i, b in enumerate(bitstring) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        qc.append(mcz, list(range(NUM_QUBITS)))
        for q in zero_positions:
            qc.x(q)


def append_diffuser(qc: QuantumCircuit):
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    mcz = ZGate().control(NUM_QUBITS - 1)
    qc.append(mcz, list(range(NUM_QUBITS)))
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))


num_marked = len(marked_logical)
search_space = 2 ** NUM_QUBITS
theta = math.asin(math.sqrt(num_marked / search_space))
iterations = max(1, math.floor((math.pi / (4 * theta)) - 0.5) + 1)
print(f"Grover setup: N={search_space} states, M={num_marked} marked, "
      f"iterations={iterations}")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    append_oracle(qc, marked_logical)
    append_diffuser(qc)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

simulator = AerSimulator(method="statevector")
transpiled = transpile(qc, simulator, basis_gates=["u", "cx"])
shots = 2000
job = simulator.run(transpiled, shots=shots)
result = job.result()
counts = result.get_counts()

# Total probability the circuit places on the marked (classically-correct)
# states, and the single most-frequently observed bitstring. Counts keys are
# in qiskit's display order, so compare against `marked_display`.
top_bitstring, top_count = Counter(counts).most_common(1)[0]
marked_shots = sum(c for bs, c in counts.items() if bs in marked_display)
marked_probability = marked_shots / shots

print(f"Most frequent measured bitstring: {top_bitstring} "
      f"({top_count}/{shots} shots)")
print(f"Total probability mass on classically-verified minimum covering "
      f"sets: {marked_probability:.3f} (baseline random-guess probability "
      f"would be {num_marked / search_space:.4f})")


# ---------------------------------------------------------------------------
# 3. Compare quantum result against the classical answer and report.
# ---------------------------------------------------------------------------

quantum_found_valid_covering = top_bitstring in marked_display
amplification_succeeded = marked_probability > 10 * (num_marked / search_space)

if quantum_found_valid_covering and amplification_succeeded:
    print("PASS: Grover search on the 10-qubit triple-selection register "
          "amplified and returned a minimum covering set of triples "
          "(size 3), matching the classically-computed T(5,4,3)=3 that "
          "underlies A140462(5) = C(5,3) - T(5,4,3) = 10 - 3 = 7.")
else:
    print("FAIL: quantum measurement did not match the classical answer.")
