"""
Erdos problem #574 (from erdosproblems.com / the manman4/erdosproblems data
set, data/problems.yaml, number: "574").

Recorded metadata for #574:
    prize: no
    informal_status: disproved (last_update 2026-03-28)
    oeis: ["possible"]
    tags: ["graph theory", "turan number"]

LIMITATION, stated honestly up front: the "oeis" field for this problem is
the literal placeholder string "possible", not a real OEIS sequence id.
There is therefore no concrete OEIS sequence to test membership/terms
against for this problem. Per the task instructions, this script does not
fabricate an OEIS id or copy a value from nowhere. Instead it uses the one
piece of real mathematical content the entry does carry -- the "turan
number" tag -- to build a genuine, honestly-derived, finite, computable
property in the same spirit as the Erdos-Stone / Turan extremal graph
theory that problem #574 sits in (Turan-type edge-maximization problems
are exactly what the "turan number" tag denotes for this data set), and
verifies it with a real Grover search circuit. This is offered as the
best honest attempt in place of a genuine OEIS-anchored test, not as a
disguised copy of #574 itself.

Classical property being tested (computed from first principles below,
not copied from any table):
    ex(n; K3), the Turan number for triangles on n = 4 labeled vertices:
    the maximum number of edges a triangle-free graph on 4 vertices can
    have. Turan's theorem gives ex(n; K3) = floor(n^2 / 4), so for n = 4
    the classical answer is floor(16/4) = 4, realized by the complete
    bipartite graph K_{2,2}.

    The script brute-forces, in plain Python, every one of the 2^6 = 64
    labeled graphs on 4 vertices (6 possible edges), checks each for
    "triangle-free" and "has exactly 4 edges", and records the exact set
    of bitstrings (edge-subsets) satisfying both. This classical search
    is the ground truth the quantum circuit is checked against.

Quantum circuit: Grover's search over the 6-qubit space of edge subsets
of K4 (one qubit per potential edge), with an oracle built directly from
the classically-computed marked set (multi-controlled-Z per marked
bitstring, i.e. a literal phase-oracle construction for "is this graph
triangle-free with exactly floor(n^2/4) edges"), followed by the standard
Grover diffuser, run on the ideal Qiskit AerSimulator. The number of
Grover iterations is chosen from the standard formula
floor(pi/4 * sqrt(N/M)) for N = 64, M = |marked set|.

PASS criterion: the bitstrings observed with high probability after
running the circuit are exactly members of the classically-computed
marked set (i.e. every outcome that has non-negligible sampled
probability is triangle-free with 4 edges), and the marked set's size
and the Turan-number value match the values computed independently by
brute force.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from itertools import combinations, product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute force over all labeled graphs on 4
#    vertices (6 possible edges), computed from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
VERTICES = list(range(N_VERTICES))
EDGES = list(combinations(VERTICES, 2))          # 6 possible edges
N_EDGES = len(EDGES)                              # 6 qubits
TRIANGLES = list(combinations(VERTICES, 3))       # 4 possible triangles

TURAN_TARGET = (N_VERTICES ** 2) // 4             # Turan's theorem: floor(n^2/4)


def edge_index(e):
    return EDGES.index(tuple(sorted(e)))


def is_triangle_free(bits):
    """bits: tuple of 0/1 of length N_EDGES, bits[i] = 1 iff EDGES[i] present."""
    present = {EDGES[i] for i in range(N_EDGES) if bits[i]}
    for (a, b, c) in TRIANGLES:
        e1 = tuple(sorted((a, b)))
        e2 = tuple(sorted((b, c)))
        e3 = tuple(sorted((a, c)))
        if e1 in present and e2 in present and e3 in present:
            return False
    return True


def brute_force_marked_set():
    """All labeled graphs on 4 vertices that are triangle-free AND have
    exactly TURAN_TARGET edges (i.e. extremal triangle-free graphs)."""
    marked = []
    max_triangle_free_edges = -1
    for bits in product([0, 1], repeat=N_EDGES):
        if is_triangle_free(bits) and sum(bits) > max_triangle_free_edges:
            # track the true max edge count among triangle-free graphs,
            # independently of the Turan formula, as a cross-check
            pass
    for bits in product([0, 1], repeat=N_EDGES):
        if is_triangle_free(bits):
            max_triangle_free_edges = max(max_triangle_free_edges, sum(bits))
    for bits in product([0, 1], repeat=N_EDGES):
        if is_triangle_free(bits) and sum(bits) == TURAN_TARGET:
            marked.append(bits)
    return marked, max_triangle_free_edges


MARKED_SET, BRUTE_FORCE_MAX_TF_EDGES = brute_force_marked_set()

assert BRUTE_FORCE_MAX_TF_EDGES == TURAN_TARGET, (
    "Turan's theorem check failed: brute-force max triangle-free edge count "
    f"({BRUTE_FORCE_MAX_TF_EDGES}) does not match floor(n^2/4) = {TURAN_TARGET}"
)
assert len(MARKED_SET) > 0, "no extremal triangle-free graphs found"

print(f"Classical ground truth: n = {N_VERTICES} vertices, {N_EDGES} possible edges.")
print(f"Turan's theorem predicts ex(4; K3) = floor(16/4) = {TURAN_TARGET}.")
print(f"Brute-force max triangle-free edge count = {BRUTE_FORCE_MAX_TF_EDGES} (matches).")
print(f"Number of extremal (4-edge, triangle-free) labeled graphs on 4 vertices: "
      f"{len(MARKED_SET)}")
for m in MARKED_SET:
    present = [EDGES[i] for i in range(N_EDGES) if m[i]]
    print(f"  marked bitstring {''.join(map(str, m))}  edges={present}")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for the marked set above.
# ---------------------------------------------------------------------------

def build_oracle(marked_bitstrings, n_qubits):
    """Phase oracle: flips the sign of exactly the basis states in
    marked_bitstrings (each a tuple of 0/1 of length n_qubits, qubit i ->
    bit i, little-endian to match Qiskit's qubit ordering)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in marked_bitstrings:
        # bits[i] refers to qubit i; flip qubits that should be 0 so the
        # target pattern becomes all-ones, apply a multi-controlled Z via
        # H-MCX-H on the last qubit, then flip back.
        zero_qubits = [i for i in range(n_qubits) if bits[i] == 0]
        for q in zero_qubits:
            qc.x(q)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


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


def grover_iterations(n_total, n_marked):
    theta = np.arcsin(np.sqrt(n_marked / n_total))
    iterations = max(1, int(np.floor((np.pi / 4) / theta - 0.5)))
    return iterations


N_TOTAL = 2 ** N_EDGES
n_iter = grover_iterations(N_TOTAL, len(MARKED_SET))
print(f"\nGrover search: N = {N_TOTAL} states, M = {len(MARKED_SET)} marked, "
      f"running {n_iter} iteration(s).")

oracle = build_oracle(MARKED_SET, N_EDGES)
diffuser = build_diffuser(N_EDGES)

qc = QuantumCircuit(N_EDGES, N_EDGES)
qc.h(range(N_EDGES))
for _ in range(n_iter):
    qc.append(oracle.to_gate(), range(N_EDGES))
    qc.append(diffuser.to_gate(), range(N_EDGES))
qc.measure(range(N_EDGES), range(N_EDGES))

simulator = AerSimulator()
compiled = transpile(qc, simulator)
shots = 4096
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()


# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

# Qiskit reports classical-register bitstrings MSB-first with qubit 0 as the
# rightmost character; reverse to index back into EDGES/qubit order.
def counts_key_to_bits(key):
    reversed_key = key[::-1]
    return tuple(int(c) for c in reversed_key)


marked_set_as_set = {bits for bits in MARKED_SET}

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("\nTop measured outcomes:")
for key, cnt in sorted_counts[:8]:
    bits = counts_key_to_bits(key)
    frac = cnt / shots
    is_marked = bits in marked_set_as_set
    print(f"  {key}  bits={bits}  count={cnt} ({frac:.3f})  "
          f"in_marked_set={is_marked}")

# Success criterion: outcomes carrying at least 5% of the shots must all be
# members of the classically-computed marked set, and the marked set must
# collectively carry most of the probability mass (Grover amplification
# worked), and the marked set / Turan number themselves check out (already
# asserted above).
significant = [(counts_key_to_bits(k), c) for k, c in counts.items()
               if c / shots >= 0.05]
all_significant_are_marked = all(bits in marked_set_as_set for bits, _ in significant)

marked_mass = sum(c for k, c in counts.items()
                   if counts_key_to_bits(k) in marked_set_as_set) / shots
print(f"\nFraction of shots landing on a marked (Turan-extremal) state: "
      f"{marked_mass:.3f}")

passed = (
    all_significant_are_marked
    and marked_mass > 0.5
    and BRUTE_FORCE_MAX_TF_EDGES == TURAN_TARGET
)

print("\nPASS" if passed else "\nFAIL")
