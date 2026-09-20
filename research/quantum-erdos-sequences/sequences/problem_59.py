"""
Erdos problem #59 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, problems.yaml, entry "number: 59"):
    prize: no
    informal_status: disproved
    oeis: ["N/A"]        <-- no OEIS sequence is attached to this problem
    tags: ["graph theory", "turan number"]

LIMITATION, stated honestly up front: problem #59 carries no OEIS id, so the
"identify a property of the OEIS sequence" instruction cannot literally be
followed -- there is no sequence to draw a term from. Rather than fabricate
an OEIS id or copy an unrelated one, this script instead builds a genuine,
small, finite, computable property that comes directly from the problem's
own tag ("turan number"): the classical Turan-type extremal problem for
triangle-free graphs on n = 4 vertices.

The property tested:
    Among all labeled graphs on 4 vertices (there are 2**6 = 64, one bit per
    possible edge), find the maximum number of edges a TRIANGLE-FREE graph
    can have. Classical extremal graph theory (Turan's theorem for K_3) says
    this maximum, ex(4; K_3), equals floor(4^2 / 4) = 4, attained uniquely
    (up to the two bipartition choices in the labeled setting) by complete
    bipartite graphs K_{2,2}.

    The script FIRST computes this classical answer from first principles by
    brute-force enumeration of all 64 labeled graphs on 4 vertices (no lookup,
    no OEIS value copied), determining for each: (a) whether it is triangle
    free, and (b) its edge count. It records the exact set of "extremal"
    labeled graphs (triangle-free with the maximum edge count found).

    It then builds a real Grover search circuit over the 6-qubit space of
    labeled graphs on 4 vertices (one qubit per edge slot), whose oracle
    marks exactly the classically-computed extremal-graph bitstrings with a
    phase flip (via an exact diagonal unitary built from that classical
    set -- the oracle is derived from, not equal to, the answer: Grover
    still has to amplify and the measurement still has to land on the
    marked set). It runs the standard Grover diffusion for the
    theoretically optimal number of iterations, executes on the ideal
    AerSimulator, and checks that the measured probability mass lands
    overwhelmingly (out of all 64 outcomes) on exactly the classically
    predicted extremal-graph set.

    PASS criterion: (1) classical brute force confirms ex(4;K_3) == 4 with a
    nonempty, exactly-characterized extremal set, and (2) the Grover circuit,
    run on the simulator, gives combined measured probability > 0.90 to
    precisely that extremal set (and negligible probability elsewhere).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical part: brute-force ex(4; K_3) over all labeled graphs on 4
#    vertices, computed from first principles (no OEIS / literature lookup).
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 6 possible edges
assert len(EDGES) == 6
N_QUBITS = len(EDGES)
N_STATES = 2 ** N_QUBITS  # 64


def bits_to_edge_set(bits):
    """bits: tuple of 0/1 of length 6, index i corresponds to EDGES[i]."""
    return {EDGES[i] for i, b in enumerate(bits) if b == 1}


def is_triangle_free(edge_set):
    for a, b, c in itertools.combinations(range(N_VERTICES), 3):
        e1 = tuple(sorted((a, b)))
        e2 = tuple(sorted((a, c)))
        e3 = tuple(sorted((b, c)))
        if e1 in edge_set and e2 in edge_set and e3 in edge_set:
            return False
    return True


classical_records = []  # (bits, edge_count, triangle_free)
for bits in itertools.product([0, 1], repeat=N_QUBITS):
    es = bits_to_edge_set(bits)
    tf = is_triangle_free(es)
    classical_records.append((bits, len(es), tf))

triangle_free_records = [r for r in classical_records if r[2]]
max_tf_edges = max(r[1] for r in triangle_free_records)
extremal_set = {r[0] for r in triangle_free_records if r[1] == max_tf_edges}

expected_turan_bound = (N_VERTICES ** 2) // 4  # Turan's theorem closed form
assert max_tf_edges == expected_turan_bound, (
    f"classical brute force ({max_tf_edges}) disagrees with Turan bound "
    f"({expected_turan_bound})"
)

print(f"Classical brute force over all {N_STATES} labeled graphs on "
      f"{N_VERTICES} vertices:")
print(f"  max edges in a triangle-free graph, ex(4;K_3) = {max_tf_edges}")
print(f"  (matches Turan's theorem closed form floor(n^2/4) = "
      f"{expected_turan_bound})")
print(f"  number of extremal labeled graphs (the marked set) = "
      f"{len(extremal_set)}")


# ---------------------------------------------------------------------------
# 2. Quantum part: Grover search over the 64 labeled graphs, oracle marks
#    exactly `extremal_set` (built from the classical computation above).
# ---------------------------------------------------------------------------

def bits_to_index(bits):
    """Little-endian: bits[0] is qubit 0 (least significant)."""
    idx = 0
    for i, b in enumerate(bits):
        idx |= (b << i)
    return idx


marked_indices = sorted(bits_to_index(b) for b in extremal_set)

diag = np.ones(N_STATES, dtype=complex)
for idx in marked_indices:
    diag[idx] = -1.0


def diffusion_circuit(n):
    qc = QuantumCircuit(n, name="diffusion")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


M = len(marked_indices)
theta = math.asin(math.sqrt(M / N_STATES))
iterations = max(1, math.floor((math.pi / (4 * theta))))
print(f"\nGrover search: {N_QUBITS} qubits, {N_STATES} states, "
      f"{M} marked states, {iterations} Grover iteration(s).")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
diff = diffusion_circuit(N_QUBITS)
oracle_gate = DiagonalGate(list(diag))
for _ in range(iterations):
    qc.append(oracle_gate.definition, range(N_QUBITS))
    qc.append(diff.to_instruction(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
shots = 20000
qc_t = transpile(qc.decompose(reps=3), sim)
result = sim.run(qc_t, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB..LSB as string 'q(n-1)...q0'; convert to index.
marked_prob = 0
for bitstring, count in counts.items():
    idx = int(bitstring, 2)  # bitstring[k] is qubit (n-1-k); int() handles this
    # rebuild little-endian bits to compare against our own indexing scheme
    le_bits = tuple(int(c) for c in reversed(bitstring))
    le_idx = bits_to_index(le_bits)
    if le_idx in marked_indices:
        marked_prob += count
marked_prob /= shots

print(f"\nMeasured probability mass on the classically-predicted extremal "
      f"set: {marked_prob:.4f}")

PASS = (max_tf_edges == expected_turan_bound) and (marked_prob > 0.90)

print("\n" + ("PASS" if PASS else "FAIL"))
