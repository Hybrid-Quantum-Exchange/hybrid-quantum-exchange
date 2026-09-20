"""
Erdos problem #579 (erdosproblems.com), quantum-testable lane.

Source metadata (data/problems.yaml, block "number: \"579\""):
    prize: no
    tags: ["graph theory", "turan number"]
    oeis: ["N/A"]

LIMITATION: problem #579 carries no OEIS sequence id in the data file
(oeis: ["N/A"]). There is therefore no OEIS sequence to build a
membership/term-search circuit against. Rather than fabricate an OEIS id
or copy a value with no real content, this script instead builds a
genuine, finite, computable instance of the exact combinatorial quantity
the problem's own tags name: a Turan number.

Classical property tested
--------------------------
Turan's theorem (the triangle case, due to Mantel) says the maximum
number of edges in a triangle-free (K3-free) simple graph on n vertices
is ex(n, K3) = floor(n^2 / 4), achieved by the complete bipartite graph
K_{ceil(n/2), floor(n/2)}.

For n = 4 vertices there are C(4,2) = 6 possible edges, so the space of
labeled simple graphs on 4 vertices has exactly 2^6 = 64 elements -- a
small, finite, fully enumerable search space, and exactly the kind of
counting/search instance a small Grover circuit can genuinely run.

This script:
  1. Enumerates all 64 labeled graphs on 4 vertices classically, checks
     each for triangles by brute force, and computes
         ex(4, K3) = max edge-count over triangle-free graphs.
     It also checks this equals floor(4^2/4) = 4 (Mantel's theorem),
     from first principles, not by asserting the formula.
  2. Builds a 6-qubit Grover search circuit whose oracle marks exactly
     the graphs that are simultaneously (a) triangle-free and (b) have
     the maximum edge count found in step 1 (i.e. the Turan-extremal
     graphs on 4 vertices -- the labeled copies of K_{2,2}).
  3. Runs the Grover circuit on the ideal AerSimulator and checks that
     the most-sampled bitstrings are exactly the marked (Turan-extremal)
     graphs, i.e. the quantum search recovers the same extremal graphs
     as the classical brute force.
  4. Prints PASS if the quantum-recovered top outcomes equal the
     classical set of Turan-extremal graphs, FAIL otherwise.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# 1. Classical computation: ex(4, K3) by brute force over all labeled
#    graphs on 4 vertices, and check it against Mantel's formula.
# ---------------------------------------------------------------------

N_VERTICES = 4
VERTEX_PAIRS = list(itertools.combinations(range(N_VERTICES), 2))  # 6 possible edges
N_EDGES_POSSIBLE = len(VERTEX_PAIRS)  # 6
assert N_EDGES_POSSIBLE == 6

TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))  # all 3-subsets


def edges_from_mask(mask: int):
    """Bit i of mask (0..63) says whether VERTEX_PAIRS[i] is an edge."""
    return {VERTEX_PAIRS[i] for i in range(N_EDGES_POSSIBLE) if (mask >> i) & 1}


def is_triangle_free(edge_set) -> bool:
    for (a, b, c) in TRIANGLES:
        e1 = tuple(sorted((a, b)))
        e2 = tuple(sorted((b, c)))
        e3 = tuple(sorted((a, c)))
        if e1 in edge_set and e2 in edge_set and e3 in edge_set:
            return False
    return True


triangle_free_masks = []
for mask in range(2 ** N_EDGES_POSSIBLE):
    es = edges_from_mask(mask)
    if is_triangle_free(es):
        triangle_free_masks.append((mask, len(es)))

max_edges = max(cnt for _, cnt in triangle_free_masks)
mantel_prediction = (N_VERTICES ** 2) // 4  # floor(n^2/4)
assert max_edges == mantel_prediction, (
    f"classical brute force ({max_edges}) disagrees with Mantel's "
    f"formula ({mantel_prediction})"
)

extremal_masks = sorted(m for m, cnt in triangle_free_masks if cnt == max_edges)

print(f"Classical result: ex(4, K3) = {max_edges} (Mantel prediction: {mantel_prediction})")
print(f"Number of Turan-extremal labeled graphs on 4 vertices: {len(extremal_masks)}")
print(f"Extremal edge-masks (0..63): {extremal_masks}")

# ---------------------------------------------------------------------
# 2. Build a Grover search circuit over the 6-bit edge-mask space whose
#    oracle marks exactly the Turan-extremal masks computed above.
# ---------------------------------------------------------------------

N_QUBITS = N_EDGES_POSSIBLE  # 6
N_STATES = 2 ** N_QUBITS      # 64
M_MARKED = len(extremal_masks)


def build_oracle(marked_masks, n_qubits):
    """Diagonal phase-flip oracle: |x> -> -|x> for x in marked_masks."""
    diag = np.ones(2 ** n_qubits, dtype=complex)
    for m in marked_masks:
        diag[m] = -1.0
    return Operator(np.diag(diag))


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


oracle_op = build_oracle(extremal_masks, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

# Optimal number of Grover iterations for M marked out of N states.
theta = math.asin(math.sqrt(M_MARKED / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.unitary(oracle_op, range(N_QUBITS), label="oracle")
    qc.append(diffuser.to_instruction(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit order: rightmost classical bit is qubit 0, so reverse to get
# the mask value with bit i = qubit i.
def bitstring_to_mask(bitstring: str) -> int:
    return int(bitstring[::-1], 2)

sorted_outcomes = sorted(counts.items(), key=lambda kv: -kv[1])
top_k = sorted_outcomes[:M_MARKED]
quantum_top_masks = sorted({bitstring_to_mask(bs) for bs, _ in top_k})

print(f"Grover iterations used: {iterations}")
print(f"Top {M_MARKED} measured bitstrings -> masks: {quantum_top_masks}")

# ---------------------------------------------------------------------
# 4. Compare quantum search result against the classical answer.
# ---------------------------------------------------------------------

verified = quantum_top_masks == extremal_masks

if verified:
    print("PASS: Grover search recovered exactly the classical Turan-extremal "
          "(K3-free, max-edge) graphs on 4 vertices.")
else:
    print("FAIL: quantum top outcomes do not match the classical extremal set.")
    print(f"  classical: {extremal_masks}")
    print(f"  quantum:   {quantum_top_masks}")
