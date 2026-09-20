"""
Erdos problem #113 (per data/problems.yaml in the manman4/erdosproblems clone,
"number: '113'") — a Turan-number / graph-theory problem, tagged
["graph theory", "turan number"], status "disproved (Lean)", with
oeis: ["N/A"].

LIMITATION, stated honestly up front: problem #113 has no OEIS sequence id
attached in the source metadata (oeis: ["N/A"]). There is therefore no
"sequence" to build a quantum-testable membership/term property from in the
sense the other lanes in this library use. Rather than fabricate a fake OEIS
id or copy an unrelated one, this script instead builds a genuine, finite,
classically-checkable instance of the actual mathematical object the
problem's tags name — a Turan-type extremal graph-counting question — and
verifies it with a real Grover search circuit. This is the "best honest
attempt" fallback the task allows when no OEIS id exists.

The classical property tested (Mantel's theorem, the n=2 case of Turan's
theorem, which is the classical ancestor of Turan-number questions):

    Among all labeled graphs on n = 4 vertices, the maximum number of edges
    in a TRIANGLE-FREE graph is floor(n^2 / 4) = 4, achieved exactly by the
    complete bipartite graphs K_{2,2} (one for each of the 3 ways to split
    4 labeled vertices into two size-2 parts).

A graph on 4 vertices has C(4,2) = 6 possible edges, so we represent each
candidate graph as a 6-bit string (one qubit per potential edge). We:

  1. Enumerate all 2^6 = 64 edge-subsets classically, check triangle-freeness
     and edge count directly (no shortcuts, no OEIS lookups), and determine
     the true maximum triangle-free edge count and the exact set of
     6-bit strings that achieve it. This is the classical ground truth.

  2. Build a Grover search circuit over 6 qubits whose oracle flags exactly
     those maximum triangle-free graphs (found step 1), and whose optimal
     number of Grover iterations is computed from the true number of marked
     states (out of the 64-dimensional search space).

  3. Run the circuit on the ideal AerSimulator, take the most frequent
     measured bitstring, and check that it decodes to a graph that is (a)
     triangle-free and (b) has the classically-determined maximum edge
     count of 4 — i.e. that Grover search actually found a correct extremal
     (Turan/Mantel-type) graph.

PASS/FAIL is decided by comparing the quantum search's top result against
the independently-computed classical answer.
"""

from itertools import combinations, product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate
import numpy as np


# ---------------------------------------------------------------------------
# Step 1: classical ground truth (Mantel's theorem instance, n = 4)
# ---------------------------------------------------------------------------

N_VERTICES = 4
VERTEX_PAIRS = list(combinations(range(N_VERTICES), 2))  # 6 possible edges
N_EDGE_BITS = len(VERTEX_PAIRS)
assert N_EDGE_BITS == 6


def bits_to_edges(bits):
    """bits: tuple of 0/1, length 6, one per entry of VERTEX_PAIRS."""
    return {VERTEX_PAIRS[i] for i, b in enumerate(bits) if b == 1}


def is_triangle_free(edges):
    for a, b, c in combinations(range(N_VERTICES), 3):
        if (a, b) in edges and (b, c) in edges and (a, c) in edges:
            return False
    return True


def bitstring_to_int_index(bits):
    """Map a 6-bit tuple to its integer value (bit i = 2**i), matching the
    little-endian qubit ordering Qiskit measurement bitstrings use."""
    return sum(b << i for i, b in enumerate(bits))


all_bitstrings = list(product([0, 1], repeat=N_EDGE_BITS))
triangle_free_edge_counts = {}
for bits in all_bitstrings:
    edges = bits_to_edges(bits)
    if is_triangle_free(edges):
        triangle_free_edge_counts[bits] = len(edges)

classical_max_edges = max(triangle_free_edge_counts.values())
mantel_bound = (N_VERTICES ** 2) // 4  # floor(n^2/4)
assert classical_max_edges == mantel_bound, (
    f"classical search found max {classical_max_edges}, "
    f"expected Mantel bound {mantel_bound}"
)

marked_bitstrings = [
    bits for bits, count in triangle_free_edge_counts.items()
    if count == classical_max_edges
]
marked_indices = sorted(bitstring_to_int_index(b) for b in marked_bitstrings)

print(f"Classical ground truth: n={N_VERTICES} vertices, "
      f"{N_EDGE_BITS}-bit edge encoding, search space size = "
      f"{2 ** N_EDGE_BITS}")
print(f"Mantel/Turan bound floor(n^2/4) = {mantel_bound}")
print(f"Max triangle-free edge count found classically = "
      f"{classical_max_edges}")
print(f"Number of maximum triangle-free graphs (marked states) = "
      f"{len(marked_bitstrings)}")
print(f"Marked computational-basis indices: {marked_indices}")


# ---------------------------------------------------------------------------
# Step 2: Grover search circuit over the 6-qubit edge-subset space
# ---------------------------------------------------------------------------

n = N_EDGE_BITS


def build_oracle(marked_bits_list):
    """Phase-flip oracle: for each marked 6-bit string, X-gate the 0-bits,
    apply a multi-controlled Z (via H + MCX + H on the last qubit), then
    undo the X-gates. Marked states are mutually orthogonal computational
    basis states, so the phase kicks compose without interference."""
    qc = QuantumCircuit(n, name="oracle")
    mcx = MCXGate(n - 1)
    for bits in marked_bits_list:
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(n - 1)
        qc.append(mcx, list(range(n - 1)) + [n - 1])
        qc.h(n - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser():
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    mcx = MCXGate(n - 1)
    qc.h(n - 1)
    qc.append(mcx, list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


num_marked = len(marked_bitstrings)
search_space = 2 ** n
# Optimal number of Grover iterations for the known number of marked states.
theta = np.arcsin(np.sqrt(num_marked / search_space))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

oracle = build_oracle(marked_bitstrings)
diffuser = build_diffuser()

qc = QuantumCircuit(n, n)
qc.h(range(n))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(n))
    qc.append(diffuser.to_gate(), range(n))
qc.measure(range(n), range(n))

print(f"Grover iterations used: {iterations}")


# ---------------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator and compare to classical ground truth
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 2048
job = backend.run(transpiled, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit returns bitstrings MSB-first (qubit n-1 ... qubit 0); convert to the
# little-endian bit tuple used above.
best_bitstring = max(counts, key=counts.get)
best_bits = tuple(int(c) for c in reversed(best_bitstring))
best_index = bitstring_to_int_index(best_bits)
best_count = counts[best_bitstring]

found_edges = bits_to_edges(best_bits)
found_triangle_free = is_triangle_free(found_edges)
found_edge_count = len(found_edges)

print(f"Top measured bitstring: {best_bitstring} "
      f"(index {best_index}, {best_count}/{shots} shots)")
print(f"Decoded edge set: {sorted(found_edges)}")
print(f"Triangle-free: {found_triangle_free}, edge count: {found_edge_count}")

verified = (
    best_index in marked_indices
    and found_triangle_free
    and found_edge_count == classical_max_edges
)

# Sanity: Grover amplification should concentrate most shots on marked states.
marked_shot_total = sum(
    counts.get(format(idx, f"0{n}b")[::-1].zfill(n)[::-1], 0)
    for idx in marked_indices
)
marked_shot_total = sum(
    count for bstr, count in counts.items()
    if bitstring_to_int_index(tuple(int(c) for c in reversed(bstr))) in marked_indices
)
amplification_ok = marked_shot_total / shots > 0.5

print(f"Fraction of shots landing on a marked (extremal) state: "
      f"{marked_shot_total}/{shots} = {marked_shot_total / shots:.3f}")

if verified and amplification_ok:
    print("PASS")
else:
    print("FAIL")
