"""
Quantum-testable instance for Erdos problem #765 (erdosproblems.com/765).

Problem metadata (data/problems.yaml): status "solved (Lean)", tags
["graph theory", "turan number"], OEIS id A006855.

A006855 is the sequence of Turan numbers ex(n, K3): the maximum number of
edges a triangle-free graph on n labeled vertices can have. By Mantel's
theorem (the n=3 case of Turan's theorem, which underlies the general Turan
number problem this OEIS sequence records) that maximum equals floor(n^2/4),
achieved exactly by the complete bipartite graphs K_{floor(n/2),ceil(n/2)}.

Classical property tested here (n = 4, the smallest non-trivial, non-complete
case where the bound is interesting):

    ex(4, K3) = max over all graphs G on 4 labeled vertices, with G
    triangle-free, of |E(G)|.

There are C(4,2) = 6 possible edges on 4 labeled vertices, hence 2^6 = 64
candidate edge-subsets. The script first brute-forces this small search space
*classically*, from first principles (no OEIS value is copied in): for every
subset of the 6 possible edges it checks all C(4,3) = 4 triples of vertices
for a triangle, keeps the triangle-free subsets, and takes the maximum edge
count among them. This reproduces a(4) = floor(16/4) = 4 from A006855, and
also records exactly which edge-subsets ("labeled graphs") attain it -- the
three copies of K_{2,2} (each omitting a distinct perfect matching from the
complete graph K4).

Quantum circuit: Grover search over the 6-qubit space of all edge-subsets of
K4 (one qubit per possible edge), with a phase oracle that marks precisely
the edge-subsets found classically to be triangle-free graphs with the
maximum edge count. This is a genuine oracle construction (multi-controlled-Z
gates targeting exactly the classically-verified marked bitstrings, built
programmatically from the classical search results -- not hand-picked), run
on the ideal AerSimulator, with the standard optimal number of Grover
iterations for N = 64, M = |marked set|. The script measures the circuit and
checks that the most frequently sampled outcome is one of the classically
verified maximum triangle-free edge-subsets, and that its edge count matches
the classically computed Turan number ex(4, K3).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles): Turan number ex(4, K3).
# ---------------------------------------------------------------------------

N_VERTICES = 4
VERTICES = list(range(N_VERTICES))
POSSIBLE_EDGES = list(itertools.combinations(VERTICES, 2))  # 6 possible edges
NUM_EDGE_SLOTS = len(POSSIBLE_EDGES)  # qubit count
TRIPLES = list(itertools.combinations(VERTICES, 3))  # 4 possible triangles


def edges_from_bits(bits):
    """bits: tuple/list of 0/1 of length NUM_EDGE_SLOTS -> set of edges present."""
    return {POSSIBLE_EDGES[i] for i, b in enumerate(bits) if b}


def is_triangle_free(edge_set):
    for a, b, c in TRIPLES:
        if (a, b) in edge_set and (b, c) in edge_set and (a, c) in edge_set:
            return False
    return True


def bits_to_bitstring(bits):
    # qiskit bit ordering: classical register c[0] is the rightmost character.
    # We build the string so that string[NUM_EDGE_SLOTS-1-i] == bits[i], i.e.
    # bits[0] is the least-significant (rightmost) character.
    return "".join(str(b) for b in reversed(bits))


triangle_free_subsets = []  # list of (bits tuple, edge_count)
for bits in itertools.product([0, 1], repeat=NUM_EDGE_SLOTS):
    es = edges_from_bits(bits)
    if is_triangle_free(es):
        triangle_free_subsets.append((bits, len(es)))

classical_max_edges = max(count for _, count in triangle_free_subsets)
marked_bits_list = [bits for bits, count in triangle_free_subsets if count == classical_max_edges]

expected_turan_number = (N_VERTICES ** 2) // 4  # Mantel/Turan closed form, floor(n^2/4)
assert classical_max_edges == expected_turan_number, (
    f"Brute-force ex(4,K3)={classical_max_edges} disagrees with floor(n^2/4)={expected_turan_number}"
)

marked_bitstrings = {bits_to_bitstring(bits) for bits in marked_bits_list}

print(f"Classical search: {len(triangle_free_subsets)} triangle-free edge-subsets out of "
      f"{2 ** NUM_EDGE_SLOTS} total subsets of K4's {NUM_EDGE_SLOTS} possible edges.")
print(f"Turan number ex(4, K3) = {classical_max_edges} "
      f"(A006855(4), matches floor(4^2/4) = {expected_turan_number}).")
print(f"Number of maximum triangle-free labeled graphs on 4 vertices: {len(marked_bits_list)} "
      f"(the three copies of K_{{2,2}}).")

# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classically-found maximum subsets.
# ---------------------------------------------------------------------------

N = 2 ** NUM_EDGE_SLOTS
M = len(marked_bits_list)


def apply_multi_controlled_z_on_bitstring(qc, bits):
    """Flip phase of the single basis state matching `bits` (tuple of 0/1,
    bits[i] is qubit i's required value), leaving all others untouched."""
    zero_positions = [i for i, b in enumerate(bits) if b == 0]
    for q in zero_positions:
        qc.x(q)
    if NUM_EDGE_SLOTS == 1:
        qc.z(0)
    else:
        qc.h(NUM_EDGE_SLOTS - 1)
        qc.mcx(list(range(NUM_EDGE_SLOTS - 1)), NUM_EDGE_SLOTS - 1)
        qc.h(NUM_EDGE_SLOTS - 1)
    for q in zero_positions:
        qc.x(q)


def build_oracle(qc):
    for bits in marked_bits_list:
        apply_multi_controlled_z_on_bitstring(qc, bits)


def build_diffuser(qc):
    qc.h(range(NUM_EDGE_SLOTS))
    qc.x(range(NUM_EDGE_SLOTS))
    qc.h(NUM_EDGE_SLOTS - 1)
    qc.mcx(list(range(NUM_EDGE_SLOTS - 1)), NUM_EDGE_SLOTS - 1)
    qc.h(NUM_EDGE_SLOTS - 1)
    qc.x(range(NUM_EDGE_SLOTS))
    qc.h(range(NUM_EDGE_SLOTS))


num_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

qc = QuantumCircuit(NUM_EDGE_SLOTS, NUM_EDGE_SLOTS)
qc.h(range(NUM_EDGE_SLOTS))
for _ in range(num_iterations):
    build_oracle(qc)
    build_diffuser(qc)
qc.measure(range(NUM_EDGE_SLOTS), range(NUM_EDGE_SLOTS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
result = backend.run(qc, shots=shots).result()
counts = result.get_counts()

most_common_bitstring, most_common_count = max(counts.items(), key=lambda kv: kv[1])
prob_marked = sum(c for bs, c in counts.items() if bs in marked_bitstrings) / shots

# Decode the measured bitstring back into an edge-subset and re-check it
# classically, exactly as a real verification of the quantum output.
measured_bits = tuple(int(ch) for ch in reversed(most_common_bitstring))
measured_edges = edges_from_bits(measured_bits)
measured_is_triangle_free = is_triangle_free(measured_edges)
measured_edge_count = len(measured_edges)

quantum_found_optimum = (
    most_common_bitstring in marked_bitstrings
    and measured_is_triangle_free
    and measured_edge_count == classical_max_edges
)

print(f"Grover search space N={N}, marked states M={M}, iterations={num_iterations}.")
print(f"Most frequent measured bitstring: {most_common_bitstring} "
      f"({most_common_count}/{shots} shots, edges={measured_edges}, "
      f"triangle-free={measured_is_triangle_free}, edge_count={measured_edge_count}).")
print(f"Total probability mass on a classically-verified maximum triangle-free graph: "
      f"{prob_marked:.3f}")

# Success requires both: (a) Grover concentrated most of its probability mass
# on the marked (optimal) states, and (b) the single most likely outcome is
# itself a valid maximum triangle-free graph matching the classical answer.
success = quantum_found_optimum and prob_marked > 0.5

if success:
    print("PASS")
else:
    print("FAIL")
