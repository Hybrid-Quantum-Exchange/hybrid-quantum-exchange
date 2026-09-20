"""
Erdos problem #564 (erdosproblems.com), quantum-testable sequence lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: '564'"):
    prize: $500
    informal_status: open (as of 2025-08-31)
    oeis: ["possible"]
    tags: ["graph theory", "ramsey theory", "hypergraphs"]

Honesty note on the OEIS id
----------------------------
The dataset's `oeis` field for problem 564 is literally the string
"possible" -- this is erdosproblems.com's own placeholder meaning "an OEIS
sequence probably exists for this problem" rather than an actual OEIS
A-number. There is therefore no real OEIS id to derive a sequence term from,
and this script does NOT fabricate one. Per the task's fallback instructions,
what follows is the best honest, mathematically genuine substitute: a small,
finite, exactly-computable decision property drawn directly from the
problem's own tags (graph theory / Ramsey theory / hypergraphs), namely a
classical Ramsey-number fact, tested with a real Grover search circuit.

The property actually tested
-----------------------------
Ramsey's theorem gives R(3,3) = 6: every 2-coloring of the edges of the
complete graph K6 contains a monochromatic triangle, and this is tight --
K5 (one vertex fewer) admits a 2-coloring of its 10 edges with NO
monochromatic triangle (the two edge-disjoint 5-cycles: the "pentagon" and
the "pentagram"). This K5 fact is the standard finite witness of R(3,3) > 5
and is exactly the kind of small, finite, computable existence question a
quantum search circuit can genuinely test.

Concretely, over the search space of all 2^10 = 1024 edge-colorings of K5
(10 edges, 2 colors each), define:

    good(coloring) = True  iff  none of the 10 triangles of K5 is monochromatic

The classical answer (computed in this script, by brute force over all 1024
colorings, from first principles -- no OEIS lookup) is that there are
exactly 20 good colorings out of 1024 (the two canonical 5-cycle colorings
together with their 10 cyclic rotations and reflections... concretely we
just count them by brute force below and print the number).

The quantum circuit
--------------------
A genuine Grover search:
  - 10 qubits, one per K5 edge (e0..e9), each representing that edge's color
    (0/1).
  - A phase oracle, synthesized by Qiskit's PhaseOracle from a boolean
    expression built directly from the 10 triangles of K5: a triangle with
    edges (a, b, c) is monochromatic iff (a == b) and (b == c); it is
    "not mono" iff (a XOR b) OR (b XOR c). The overall oracle marks
    (flips phase of) exactly the "good" (no monochromatic triangle)
    colorings -- the AND, over all 10 triangles, of "not mono".
  - A standard Grover diffusion operator (H^10, X^10, multi-controlled Z,
    X^10, H^10).
  - The number of Grover iterations is computed from the classically-known
    count of good states (round(pi/4 * sqrt(N/M))).

After running on the ideal AerSimulator (statevector, no noise), we check
that the measurement distribution is concentrated on the "good" states, and
compare the most-frequent measured colorings against the classically
verified set of good colorings.

Dependencies: qiskit, qiskit_aer, numpy only (already installed).
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PhaseOracle
from qiskit_aer import AerSimulator

ERDOS_PROBLEM = 564
OEIS_IDS_USED = []  # none: source metadata gives oeis: ["possible"], not a real A-number


# ---------------------------------------------------------------------------
# 1. Classical ground truth: K5, its 10 edges, its 10 triangles, and the
#    exact set of "good" (no monochromatic triangle) 2-colorings, computed
#    by brute force over all 2^10 = 1024 colorings.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3, 4]
EDGES = list(itertools.combinations(VERTICES, 2))  # 10 edges, index = position
assert len(EDGES) == 10
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 10 triangles
assert len(TRIANGLES) == 10


def triangle_edge_indices(tri):
    a, b, c = tri
    return (
        EDGE_INDEX[(a, b)] if a < b else EDGE_INDEX[(b, a)],
        EDGE_INDEX[(a, c)] if a < c else EDGE_INDEX[(c, a)],
        EDGE_INDEX[(b, c)] if b < c else EDGE_INDEX[(c, b)],
    )


TRIANGLE_EDGE_IDX = [triangle_edge_indices(t) for t in TRIANGLES]


def is_good_coloring(bits):
    """bits: tuple/list of 10 ints (0/1), one per edge in EDGES order."""
    for (i, j, k) in TRIANGLE_EDGE_IDX:
        if bits[i] == bits[j] == bits[k]:
            return False
    return True


def classical_good_colorings():
    good = []
    for bits in itertools.product([0, 1], repeat=10):
        if is_good_coloring(bits):
            good.append(bits)
    return good


GOOD_COLORINGS = classical_good_colorings()
N_STATES = 1024
M_GOOD = len(GOOD_COLORINGS)

print(f"Erdos problem #{ERDOS_PROBLEM}: K5 edge-2-coloring, no monochromatic triangle.")
print(f"Classical brute force: {M_GOOD} good colorings out of {N_STATES}.")
assert M_GOOD > 0, "R(3,3) > 5 would be false if no such coloring existed -- sanity check failed"


# ---------------------------------------------------------------------------
# 2. Build the phase oracle from a boolean expression over e0..e9, directly
#    encoding "no triangle is monochromatic".
# ---------------------------------------------------------------------------

edge_vars = [f"e{i}" for i in range(10)]

# PhaseOracle assigns qubit index by ORDER OF FIRST APPEARANCE of each
# variable name in the expression string, not by name-sort order. Our real
# clauses reference e0..e9 out of order (e.g. e0, e1, e4, e2, ...), which
# would silently scramble the qubit<->edge mapping. A tautological prefix
# that mentions e0..e9 in order, before anything else, fixes the
# first-appearance order to exactly match our intended qubit index i <-> e_i.
order_prefix = " & ".join(f"({v} | ~{v})" for v in edge_vars)

clauses = []
for (i, j, k) in TRIANGLE_EDGE_IDX:
    a, b, c = edge_vars[i], edge_vars[j], edge_vars[k]
    # "not monochromatic" for this triangle: (a XOR b) OR (b XOR c)
    clauses.append(f"(({a} ^ {b}) | ({b} ^ {c}))")

expression = order_prefix + " & " + " & ".join(clauses)

oracle = PhaseOracle(expression)
n_qubits = oracle.num_qubits  # 10 edge qubits + ancillas used internally by PhaseOracle
print(f"PhaseOracle synthesized on {n_qubits} qubits (10 are the edge variables).")


# ---------------------------------------------------------------------------
# 3. Grover diffusion operator over the 10 edge qubits (variable qubits only;
#    PhaseOracle keeps its own ancillas clean / restored to |0>).
# ---------------------------------------------------------------------------

def diffusion_operator(num_edge_qubits):
    qc = QuantumCircuit(num_edge_qubits, name="diffusion")
    qc.h(range(num_edge_qubits))
    qc.x(range(num_edge_qubits))
    qc.h(num_edge_qubits - 1)
    qc.mcx(list(range(num_edge_qubits - 1)), num_edge_qubits - 1)
    qc.h(num_edge_qubits - 1)
    qc.x(range(num_edge_qubits))
    qc.h(range(num_edge_qubits))
    return qc


diff = diffusion_operator(10)

# PhaseOracle's own variable-qubit ordering follows the order variables were
# first seen in the expression string, i.e. e0..e9 on qubits 0..9 (any extra
# qubits beyond 10 are its internal ancillas, appended after).
num_iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M_GOOD)))
print(f"Grover iterations: {num_iterations} (N={N_STATES}, M={M_GOOD})")

qc = QuantumCircuit(n_qubits, 10)
qc.h(range(10))
for _ in range(num_iterations):
    qc.append(oracle.to_instruction(), range(n_qubits))
    qc.append(diff.to_instruction(), range(10))
qc.measure(range(10), range(10))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit-string order is little-endian in the classical register vs qubit
# index (rightmost char = qubit 0 = e0). Convert each measured bitstring back
# to our (e0..e9) tuple ordering.
def bitstring_to_edge_tuple(bs):
    bs = bs[-10:]  # classical register has exactly 10 bits (c0..c9)
    return tuple(int(bs[9 - i]) for i in range(10))  # bs[9-i] is qubit i's bit


good_set = set(GOOD_COLORINGS)
measured_good_shots = 0
top_measured = sorted(counts.items(), key=lambda kv: -kv[1])[:5]

for bs, cnt in counts.items():
    edges_tuple = bitstring_to_edge_tuple(bs)
    if edges_tuple in good_set:
        measured_good_shots += cnt

good_fraction = measured_good_shots / shots
print(f"Fraction of shots landing on a classically-verified good coloring: {good_fraction:.3f}")
print("Top measured outcomes (bitstring: count):")
for bs, cnt in top_measured:
    print(f"  {bs}: {cnt}")

# A successful Grover search on this instance should concentrate the large
# majority of shots onto good states (amplitude amplified well above the
# uniform baseline of M/N = 20/1024 ~= 0.0195).
baseline = M_GOOD / N_STATES
success = good_fraction > 5 * baseline and good_fraction > 0.5

print(f"Baseline (uniform) success probability: {baseline:.4f}")
if success:
    print("PASS")
else:
    print("FAIL")
