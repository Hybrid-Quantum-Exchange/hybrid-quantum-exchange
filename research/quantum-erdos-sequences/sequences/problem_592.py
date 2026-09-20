"""
Erdos problem #592 -- quantum-testable sequence lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: '592'"):
    prize: $1000
    status: open
    tags: ["set theory", "ramsey theory"]
    oeis: ["N/A"]

HONEST LIMITATION: Erdos problem #592 has **no OEIS sequence id** attached in the
source data (oeis: ["N/A"]). The task requires deriving a small, finite,
computable property from "its OEIS sequence id(s) and tags" -- there is no OEIS
id to derive one from, so nothing here is a genuine encoding of problem #592
itself or of any sequence indexed under it. Fabricating an OEIS-sourced value
would violate the no-fabrication instruction, so instead this script builds its
best honest, self-contained substitute: a real, classically-verified instance of
the one concrete finite structure named directly by the problem's own tags
("set theory", "ramsey theory") -- a genuine Ramsey-triangle search that is
finite, computable, and small enough for an ideal quantum simulator.

Classical property under test
------------------------------
For the complete graph K4 (4 vertices, C(4,2) = 6 edges), 2-color each edge
red/blue. A coloring is called "triangle-good" if none of K4's 4 triangles is
monochromatic (all 3 of its edges the same color). This is a finite classical
question about the smallest Ramsey number R(3,3) = 6: K4 (n=4 < 6) admits at
least one triangle-good coloring, while K6 does not. The script:
  1. Enumerates all 2^6 = 64 edge colorings classically and computes the exact
     set of triangle-good colorings (first principles, no OEIS lookup).
  2. Builds a Grover search circuit over the 6 qubits (one per edge) whose
     oracle marks exactly the triangle-good colorings, and runs the optimal
     number of Grover iterations on the ideal AerSimulator statevector.
  3. Confirms the quantum search amplifies the classically-known marked set,
     then samples and checks that the returned coloring is genuinely
     triangle-good by re-checking it classically.

PASS/FAIL is decided by comparing the quantum measurement outcome against the
classical brute-force truth, not by copying any external value.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no external data).
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges, bit i <-> EDGES[i]
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles
N_BITS = len(EDGES)
assert N_BITS == 6

EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}


def triangle_edge_bits(tri):
    a, b, c = tri
    e1 = tuple(sorted((a, b)))
    e2 = tuple(sorted((b, c)))
    e3 = tuple(sorted((a, c)))
    return EDGE_INDEX[e1], EDGE_INDEX[e2], EDGE_INDEX[e3]


TRIANGLE_BITS = [triangle_edge_bits(t) for t in TRIANGLES]


def is_triangle_good(bits):
    """bits: tuple of 6 ints (0/1), bit i = color of EDGES[i]. True if no
    monochromatic triangle among K4's 4 triangles."""
    for i, j, k in TRIANGLE_BITS:
        if bits[i] == bits[j] == bits[k]:
            return False
    return True


classical_marked = []
for n in range(2 ** N_BITS):
    bits = tuple((n >> b) & 1 for b in range(N_BITS))
    if is_triangle_good(bits):
        classical_marked.append(n)

classical_marked_set = set(classical_marked)
M = len(classical_marked_set)
N = 2 ** N_BITS

print(f"Classical brute force: {M} of {N} edge-colorings of K4 are triangle-good.")
assert M > 0, "R(3,3)=6 guarantees K4 has triangle-good colorings; sanity check failed"

# ---------------------------------------------------------------------------
# 2. Grover search circuit marking exactly classical_marked_set.
# ---------------------------------------------------------------------------


def build_oracle(n_bits, marked_states):
    qc = QuantumCircuit(n_bits, name="oracle")
    for state in marked_states:
        bits = [(state >> b) & 1 for b in range(n_bits)]
        zero_positions = [b for b in range(n_bits) if bits[b] == 0]
        for b in zero_positions:
            qc.x(b)
        # multi-controlled Z on all n_bits qubits (phase flip if all |1>)
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
        for b in zero_positions:
            qc.x(b)
    return qc


def build_diffuser(n_bits):
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


oracle = build_oracle(N_BITS, classical_marked_set)
diffuser = build_diffuser(N_BITS)

# Optimal number of Grover iterations for N states, M marked.
theta = np.arcsin(np.sqrt(M / N))
n_iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
print(f"Grover iterations: {n_iterations} (M={M}, N={N})")

qc = QuantumCircuit(N_BITS, N_BITS)
qc.h(range(N_BITS))
for _ in range(n_iterations):
    qc.append(oracle.to_instruction(), range(N_BITS))
    qc.append(diffuser.to_instruction(), range(N_BITS))
qc.measure(range(N_BITS), range(N_BITS))

# ---------------------------------------------------------------------------
# 3. Run on ideal AerSimulator, verify against classical truth.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 2000
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: classical bit c[0] is the rightmost character.
def outcome_to_int(bitstring):
    return int(bitstring[::-1], 2)

hits_in_marked = 0
total = 0
best_bitstring, best_count = max(counts.items(), key=lambda kv: kv[1])
best_int = outcome_to_int(best_bitstring)

for bitstring, cnt in counts.items():
    total += cnt
    if outcome_to_int(bitstring) in classical_marked_set:
        hits_in_marked += cnt

fraction_marked = hits_in_marked / total
print(f"Most frequent outcome: {best_bitstring} -> state {best_int}, "
      f"count {best_count}/{shots}")
print(f"Fraction of shots landing on a classically triangle-good coloring: "
      f"{fraction_marked:.4f}")

# Re-verify the top measured outcome classically, from first principles.
top_bits = tuple((best_int >> b) & 1 for b in range(N_BITS))
top_is_good_classically = is_triangle_good(top_bits)

# Grover success criterion: amplification should push a large majority of
# shots onto marked (triangle-good) states, and the top outcome itself must
# independently check out against the classical definition.
success = fraction_marked > 0.5 and top_is_good_classically

if success:
    print("PASS")
else:
    print("FAIL")

sys.exit(0 if success else 1)
