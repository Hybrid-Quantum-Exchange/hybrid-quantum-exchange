"""
Erdos problem #642 -- quantum-testable instance
=================================================

Source metadata (from erdosproblems.com, data/problems.yaml, entry "642"):
    prize: no
    status: open (informal and formal both open, last_update 2025-08-31)
    oeis: ["possible"]
    tags: ["graph theory", "cycles"]

Honest limitation up front: problem #642's YAML record does NOT carry a real
OEIS sequence id -- the `oeis` field is the literal placeholder string
"possible" (erdosproblems.com's own marker meaning "an OEIS sequence might
exist for this problem, but none has been linked yet"), not an A-number. So
there is no concrete integer sequence from OEIS to test membership/terms of
for this problem. Rather than fabricate an OEIS id or copy a number with no
real connection to #642, this script instead builds a genuine, small,
finite, classically-checkable decision problem drawn directly from problem
642's own two tags -- "graph theory" and "cycles" -- namely:

    Classical property tested
    --------------------------
    Fix the 4-vertex graph G = K4 minus the edge {1,3} (vertices 0,1,2,3;
    edges {0,1},{0,2},{0,3},{1,2},{2,3} -- 5 of the 6 possible edges).
    Search space: all Hamiltonian cycles of K4 starting/ending at vertex 0,
    i.e. all permutations (a,b,c) of (1,2,3), giving the cycle
    0-a-b-c-0. There are 3! = 6 such permutations, indexed 0..5, encoded
    on 3 qubits (index space 0..7, with 6 and 7 unused/never marked).
    A permutation is a valid Hamiltonian cycle of G iff all four of its
    edges {0,a},{a,b},{b,c},{c,0} are present in G.

    This script first computes, purely classically (brute force over all
    6 permutations, from first principles, no OEIS lookup), the exact set
    of indices that correspond to genuine Hamiltonian cycles of G. It then
    builds a Grover search circuit whose oracle marks exactly those
    indices, runs it on the ideal AerSimulator, and checks that the most
    frequently measured basis state is one of the classically-verified
    marked indices (i.e. Grover search finds a real Hamiltonian cycle of
    the graph). This is a real combinatorial search over "cycles" in
    "graph theory" -- the two tags problem #642 itself carries -- even
    though no OEIS sequence id could be honestly attached to it.

Circuit: standard 3-qubit Grover search (oracle = multi-controlled Z over
the marked bitstrings via X-sandwiching, diffuser = standard
inversion-about-the-mean), run on qiskit_aer's ideal AerSimulator.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (from first principles, no OEIS / external data)
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
# K4 minus edge {1,3}
EDGES = {(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)}


def has_edge(u, v):
    return (u, v) in EDGES or (v, u) in EDGES


# All permutations of (1,2,3) -> candidate Hamiltonian cycles 0-a-b-c-0
perms = list(itertools.permutations([1, 2, 3]))  # length 6, index 0..5
assert len(perms) == 6

marked_indices = []
for idx, (a, b, c) in enumerate(perms):
    cycle_edges = [(0, a), (a, b), (b, c), (c, 0)]
    if all(has_edge(u, v) for u, v in cycle_edges):
        marked_indices.append(idx)

print("Graph G = K4 minus edge {1,3}, edges:", sorted(EDGES))
print("Permutation index -> (a,b,c):")
for idx, p in enumerate(perms):
    print(f"  {idx} (bits {idx:03b}): {p}  valid_cycle={idx in marked_indices}")
print("Classically verified marked (valid Hamiltonian cycle) indices:", marked_indices)

assert len(marked_indices) > 0, "expected at least one Hamiltonian cycle in G"
N = 8  # 3-qubit search space (indices 0..7; 6,7 are padding, never marked)
M = len(marked_indices)


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 3-qubit index space
# ---------------------------------------------------------------------------

NUM_QUBITS = 3


def mark_state(qc, index, num_qubits):
    """Apply a multi-controlled Z that flips the phase of |index>."""
    bits = format(index, f"0{num_qubits}b")
    # Flip qubits that should be 0 so the target pattern becomes all-1s.
    for i, b in enumerate(reversed(bits)):
        if b == "0":
            qc.x(i)
    if num_qubits == 1:
        qc.z(0)
    elif num_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    for i, b in enumerate(reversed(bits)):
        if b == "0":
            qc.x(i)


def oracle(qc, indices, num_qubits):
    for idx in indices:
        mark_state(qc, idx, num_qubits)


def diffuser(qc, num_qubits):
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))


# Optimal number of Grover iterations for N states, M marked.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    oracle(qc, marked_indices, NUM_QUBITS)
    diffuser(qc, NUM_QUBITS)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
result = sim.run(compiled, shots=4096).result()
counts = result.get_counts()

# Qiskit bit order is little-endian in the classical register string; the
# measured bitstring's rightmost char is qubit 0. Our mark_state/diffuser
# use the same convention (bit i <-> qubit i), so we must reverse the
# printed string to get back the integer index consistent with `format`.
def bitstring_to_index(bs):
    return int(bs[::-1], 2)

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_bitstring, top_count = sorted_counts[0]
top_index = bitstring_to_index(top_bitstring)

print("\nGrover search results (top outcomes):")
for bs, cnt in sorted_counts[:5]:
    print(f"  index {bitstring_to_index(bs)} (bits {bs}): {cnt} shots")

print(f"\nMost frequent measured index: {top_index} ({top_count}/4096 shots)")
print(f"Classically valid Hamiltonian-cycle indices: {marked_indices}")

quantum_found_valid_cycle = top_index in marked_indices

# Also check that the aggregate probability mass landed mostly on marked
# states, as a sanity check that Grover actually amplified them.
marked_mass = sum(cnt for bs, cnt in counts.items() if bitstring_to_index(bs) in marked_indices)
marked_fraction = marked_mass / 4096

print(f"Fraction of shots landing on a classically-valid cycle index: {marked_fraction:.3f}")

verified = quantum_found_valid_cycle and marked_fraction > 0.5

if verified:
    print("\nPASS: Grover search's most likely outcome is a classically-verified "
          "Hamiltonian cycle of G, and most shots land on valid-cycle indices.")
else:
    print("\nFAIL: Grover search result does not match the classical answer.")

assert verified
