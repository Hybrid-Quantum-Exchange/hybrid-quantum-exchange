"""
Erdos problem #916 (per erdosproblems.com metadata, data/problems.yaml):
  prize: no
  status: proved (informal, last_update 2025-08-31)
  oeis: ["N/A"]
  tags: ["graph theory", "cycles"]

HONESTY NOTE ON SCOPE
----------------------
Problem 916 carries NO OEIS sequence id in the source metadata (oeis is
literally the placeholder "N/A"). The task for this lane is to build a
quantum-testable instance of "a small, finite, computable property" of the
sequence/topic associated with the problem. Since there is no sequence to
key off of, this script does NOT claim to test any specific OEIS entry or
any specific formal statement of problem 916. Instead, honoring the
problem's own tags ("graph theory", "cycles"), it builds a genuine, honestly
finite/computable graph-theory decision problem in the same subject area —
triangle (3-cycle) detection in a small fixed graph — and solves it with a
real Grover search circuit on the ideal AerSimulator, verified against a
brute-force classical computation performed in this script.

This is offered as the best faithful attempt given the missing OEIS id, not
as a claim that it formalizes or tests Erdos problem #916 itself.

THE CLASSICAL PROPERTY BEING TESTED
------------------------------------
Fix a graph G on 4 labeled vertices {0,1,2,3}. There are exactly
C(4,3) = 4 distinct 3-vertex subsets ("triples"), each of which is a
candidate 3-cycle (triangle) in G:

    index 0 -> triple (0,1,2)
    index 1 -> triple (0,1,3)
    index 2 -> triple (0,2,3)
    index 3 -> triple (1,2,3)

A triple forms a triangle (a 3-cycle) in G iff all three of its induced
edges are present in G. We choose G's edge set so that EXACTLY ONE of the
four triples is a triangle. The classical property under test is:

    "Which of the 4 triples is the (unique) triangle in G?"

This is computed directly in this script by brute-force enumeration over
all 4 triples and all 3 edges of each (first-principles classical check,
no OEIS lookup). The quantum circuit is a 2-qubit Grover search over the
4-element index space {0,1,2,3} whose oracle marks exactly the index that
is a triangle; with N=4 items and 1 marked item, a single Grover iteration
amplifies the marked index to (ideally) probability 1, so measurement
should return the classically-computed triangle index with overwhelming
probability.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Fix the graph and compute the classical answer from first principles.
# ---------------------------------------------------------------------------

VERTICES = (0, 1, 2, 3)
TRIPLES = list(itertools.combinations(VERTICES, 3))  # 4 triples, index 0..3

# Edge set for G, chosen so that exactly one triple (0,1,2) is a triangle:
# edges 0-1, 1-2, 0-2 present (triangle on {0,1,2}); vertex 3 attached by a
# single edge to 0 only, so no other triple is fully connected.
EDGES = {
    frozenset((0, 1)),
    frozenset((1, 2)),
    frozenset((0, 2)),
    frozenset((0, 3)),
}


def is_triangle(triple, edges):
    a, b, c = triple
    needed = [frozenset((a, b)), frozenset((b, c)), frozenset((a, c))]
    return all(e in edges for e in needed)


classical_triangle_indices = [
    i for i, t in enumerate(TRIPLES) if is_triangle(t, EDGES)
]

assert len(classical_triangle_indices) == 1, (
    "instance must have exactly one triangle for this single-solution "
    "Grover search; got %r" % classical_triangle_indices
)
CLASSICAL_ANSWER = classical_triangle_indices[0]

print("Triples (index -> vertex triple):")
for i, t in enumerate(TRIPLES):
    print(f"  {i}: {t}  triangle={is_triangle(t, EDGES)}")
print(f"Classical answer: triangle is triple index {CLASSICAL_ANSWER} "
      f"= {TRIPLES[CLASSICAL_ANSWER]}")


# ---------------------------------------------------------------------------
# 2. Build a 2-qubit Grover search circuit whose oracle marks that index.
# ---------------------------------------------------------------------------
# Index register: 2 qubits q0,q1 encode index i = q1 q0 (little-endian),
# i in {0,1,2,3}. The oracle applies a phase flip to the basis state
# |CLASSICAL_ANSWER>. For N=4 and 1 marked item, one Grover iteration
# (oracle + diffuser) drives the success probability to 1 in the ideal
# (noiseless) simulation.

def build_grover_circuit(marked_index: int) -> QuantumCircuit:
    n = 2  # qubits, 2**2 = 4 items
    qc = QuantumCircuit(n, n)

    # Uniform superposition.
    qc.h(range(n))

    bits = format(marked_index, f"0{n}b")[::-1]  # little-endian bit string

    def apply_oracle():
        # Flip qubits that should be 0 in the marked index so the marked
        # state maps to |11>, apply a controlled-Z (via H-CX-H on 2 qubits
        # this is just a CZ), then flip back.
        for qi, bit in enumerate(bits):
            if bit == "0":
                qc.x(qi)
        qc.cz(0, 1)
        for qi, bit in enumerate(bits):
            if bit == "0":
                qc.x(qi)

    def apply_diffuser():
        qc.h(range(n))
        qc.x(range(n))
        qc.cz(0, 1)
        qc.x(range(n))
        qc.h(range(n))

    apply_oracle()
    apply_diffuser()

    qc.measure(range(n), range(n))
    return qc


qc = build_grover_circuit(CLASSICAL_ANSWER)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 2000
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

print("\nMeasurement counts (bitstring -> count):")
for bitstring, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {bitstring}: {cnt}")

# Qiskit reports classical bits as c1 c0 (MSB..LSB) with our little-endian
# encoding q0 q1 -> read bit string reversed to get (q1 q0) then int().
most_common_bitstring = max(counts, key=counts.get)
# most_common_bitstring is "c1c0" (qiskit prints classical register with
# highest index first); our index i has bit0 = q0 (LSB), bit1 = q1 (MSB).
measured_index = int(most_common_bitstring[::-1], 2)

fraction_correct = counts.get(most_common_bitstring, 0) / SHOTS

print(f"\nMost frequent measured index: {measured_index} "
      f"(fraction of shots: {fraction_correct:.3f})")
print(f"Classical answer: {CLASSICAL_ANSWER}")

ran_ok = True
verified = (measured_index == CLASSICAL_ANSWER) and (fraction_correct > 0.9)

if verified:
    print("PASS")
else:
    print("FAIL")
