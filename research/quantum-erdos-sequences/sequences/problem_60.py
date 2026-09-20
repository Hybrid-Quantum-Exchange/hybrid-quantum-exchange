"""
Erdos problem #60 -- quantum-testable instance
================================================

Erdos problem #60 (data/problems.yaml, erdosproblems repo, entry "number: 60")
concerns OEIS sequence A006855: the maximum number of edges ex(n; C4) in a
graph on n labeled vertices that contains no 4-cycle (C4) as a subgraph.
Erdos's problem asks about the growth rate / exact values of this
"Zarankiewicz-type" extremal function. Tags in the source data: ["graph
theory", "cycles"].

Classical property tested here (finite, computable, and checked from first
principles in this script -- not copied from OEIS):

    For n = 4 vertices, what is the maximum number of edges a simple graph
    can have while containing no 4-cycle, and does at least one such
    maximum graph exist?

K4 has C(4,2) = 6 possible edges, indexed q0..q5:
    q0 = (1,2)   q1 = (1,3)   q2 = (1,4)
    q3 = (2,3)   q4 = (2,4)   q5 = (3,4)

There are exactly three distinct 4-cycles on 4 labeled vertices (a graph on
4 vertices with 4 edges forming a cycle visiting all vertices once):
    C1 = {(1,2),(2,3),(3,4),(4,1)} = {q0,q3,q5,q2}
    C2 = {(1,2),(2,4),(4,3),(3,1)} = {q0,q4,q5,q1}
    C3 = {(1,3),(3,2),(2,4),(4,1)} = {q1,q3,q4,q2}

A 6-bit edge-subset is C4-free iff it does not contain all 4 edges of C1,
C2, or C3 simultaneously.

This script:
  1. Classically brute-forces all 2^6 = 64 edge subsets of K4, finds every
     C4-free subset with exactly 4 edges (the classical extremal answer,
     which must match A006855's known value a(4) = 4 -- verified, not
     assumed), and records the exact set of "good" 6-bit strings.
  2. Builds a Grover search circuit over the 6 edge-qubits whose oracle
     marks precisely those "good" strings (computed independently inside
     the oracle-construction code from the same C4/edge-count logic, then
     cross-checked bit-for-bit against the brute-force set before any
     quantum circuit is built).
  3. Runs one Grover iteration (optimal for this marked-fraction) on the
     ideal AerSimulator and checks that measurement probability
     concentrates on the "good" (max-edges, C4-free) states.
  4. Prints PASS if the quantum search's most-probable outcomes are exactly
     the classically-verified good states (with amplified probability),
     FAIL otherwise.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

N_EDGES = 6
EDGES = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

# The three 4-cycles on vertices {1,2,3,4}, each as a tuple of edge indices.
CYCLES = [
    [(1, 2), (2, 3), (3, 4), (4, 1)],
    [(1, 2), (2, 4), (4, 3), (3, 1)],
    [(1, 3), (3, 2), (2, 4), (4, 1)],
]


def norm_edge(a, b):
    return (a, b) if a < b else (b, a)


CYCLE_BITS = []
for cyc in CYCLES:
    bits = [EDGE_INDEX[norm_edge(a, b)] for a, b in cyc]
    assert len(set(bits)) == 4
    CYCLE_BITS.append(sorted(bits))

print("Edge index map:", {e: i for e, i in EDGE_INDEX.items()})
print("4-cycle bit-index groups:", CYCLE_BITS)


def is_c4_free(bits_set):
    """bits_set: set of edge indices present. True if no full 4-cycle."""
    for group in CYCLE_BITS:
        if all(b in bits_set for b in group):
            return False
    return True


# --- Step 1: classical brute force over all 64 edge subsets ---------------
best_size = -1
good_states = []  # list of 6-bit strings (q5 q4 q3 q2 q1 q0 order for Qiskit)

for mask in range(2 ** N_EDGES):
    bits_set = {i for i in range(N_EDGES) if (mask >> i) & 1}
    if not is_c4_free(bits_set):
        continue
    size = len(bits_set)
    if size > best_size:
        best_size = size
        good_states = []
    if size == best_size:
        good_states.append(mask)

print(f"Classical brute force: max C4-free edge count on K4 = {best_size}")
print(f"Number of maximum C4-free edge subsets: {len(good_states)}")
assert best_size == 4, "classical result must equal OEIS A006855 a(4)=4"

good_bitstrings = sorted(
    format(m, f"0{N_EDGES}b")[::-1] for m in good_states  # little-endian q0..q5
)
print("Good (max-edge, C4-free) 6-bit states (q0 q1 q2 q3 q4 q5):", good_bitstrings)

CLASSICAL_ANSWER = {
    "max_edges": best_size,
    "num_optimal_graphs": len(good_states),
    "good_masks": sorted(good_states),
}

# --- Step 2: build the Grover oracle ---------------------------------------
# The oracle must flip the phase of exactly the "good_states" computed above.
# We build it from the *same* logical predicate (edge-count == 4 AND none of
# the three 4-cycles fully present), re-derived independently here via an
# ancilla-based multi-controlled circuit, then cross-check its marked set
# against good_states before running anything on the simulator.


def build_oracle():
    qc = QuantumCircuit(N_EDGES + 4, name="oracle")  # 6 edge qubits + 4 ancillas
    edge_q = list(range(N_EDGES))
    cyc_anc = [N_EDGES, N_EDGES + 1, N_EDGES + 2]  # one per cycle: 1 if cycle fully present
    phase_anc = N_EDGES + 3

    # mark cyc_anc[i] = 1 iff all 4 edges of cycle i are set
    for i, group in enumerate(CYCLE_BITS):
        qc.append(MCXGate(4), [edge_q[b] for b in group] + [cyc_anc[i]])

    # We want to flip phase iff: popcount(edge_q) == 4 AND all cyc_anc == 0.
    # Popcount == 4 out of 6 qubits: enumerate the C(6,4)=15 matching bit
    # patterns directly (small enough to do exactly), each triggering a flip
    # via a multi-controlled Z conditioned on (that exact 4-of-6 pattern AND
    # all three cyc_anc ancillas being 0).
    for combo in itertools.combinations(range(N_EDGES), 4):
        zero_bits = [b for b in range(N_EDGES) if b not in combo]
        # Flip X on the zero-bits so "all controls = 1" <=> this exact pattern
        for b in zero_bits:
            qc.x(edge_q[b])
        # also require all three cycle-ancillas = 0 -> X them so control=1 means anc=0
        for a in cyc_anc:
            qc.x(a)
        controls = [edge_q[b] for b in range(N_EDGES)] + cyc_anc
        qc.append(MCXGate(len(controls)), controls + [phase_anc])
        for a in cyc_anc:
            qc.x(a)
        for b in zero_bits:
            qc.x(edge_q[b])

    # phase_anc now holds 1 exactly on "good" computational basis states,
    # after starting each time from |0>; but repeated appends above OR the
    # conditions together only if phase_anc starts at 0 and only one combo
    # can ever match (edge-subsets have a unique popcount), so this is safe.
    qc.z(phase_anc)

    # uncompute (mirror the same operations in reverse to reset ancillas)
    for combo in reversed(list(itertools.combinations(range(N_EDGES), 4))):
        zero_bits = [b for b in range(N_EDGES) if b not in combo]
        for b in zero_bits:
            qc.x(edge_q[b])
        for a in cyc_anc:
            qc.x(a)
        controls = [edge_q[b] for b in range(N_EDGES)] + cyc_anc
        qc.append(MCXGate(len(controls)), controls + [phase_anc])
        for a in cyc_anc:
            qc.x(a)
        for b in zero_bits:
            qc.x(edge_q[b])

    for i, group in enumerate(CYCLE_BITS):
        qc.append(MCXGate(4), [edge_q[b] for b in group] + [cyc_anc[i]])

    return qc


oracle = build_oracle()

# Cross-check the oracle's marked set against good_states via statevector
# simulation of the oracle alone (starting from an equal superposition would
# be wasteful to inspect bit-by-bit; instead directly verify, for every one
# of the 64 basis states, that the oracle flips sign iff it is in good_states).
sim_check = AerSimulator(method="statevector")
mismatches = 0
for mask in range(2 ** N_EDGES):
    qc = QuantumCircuit(N_EDGES + 4)
    for i in range(N_EDGES):
        if (mask >> i) & 1:
            qc.x(i)
    qc.compose(oracle, inplace=True)
    qc.save_statevector()
    tqc = transpile(qc, sim_check)
    result = sim_check.run(tqc).result()
    sv = np.asarray(result.get_statevector())
    idx = mask  # ancillas are back to 0, so amplitude sits at index `mask`
    amp = sv[idx]
    flipped = np.isclose(amp, -1, atol=1e-6)
    should_flip = mask in good_states
    if flipped != should_flip:
        mismatches += 1

print(f"Oracle self-check mismatches out of 64 basis states: {mismatches}")
assert mismatches == 0, "oracle does not mark exactly the classically-verified good states"

# --- Step 3: Grover diffuser + one iteration on N_EDGES qubits -------------
grover = QuantumCircuit(N_EDGES + 4, N_EDGES)
grover.h(range(N_EDGES))

# oracle
grover.compose(oracle, inplace=True)

# diffuser on the 6 edge qubits (ancillas stay |0>)
grover.h(range(N_EDGES))
grover.x(range(N_EDGES))
grover.h(N_EDGES - 1)
grover.append(MCXGate(N_EDGES - 1), list(range(N_EDGES - 1)) + [N_EDGES - 1])
grover.h(N_EDGES - 1)
grover.x(range(N_EDGES))
grover.h(range(N_EDGES))

grover.measure(range(N_EDGES), range(N_EDGES))

sim = AerSimulator()
tqc = transpile(grover, sim)
job = sim.run(tqc, shots=4096)
counts = job.result().get_counts()

# Qiskit bit order in counts strings is q_{n-1}...q_0; convert to our
# little-endian (q0 first) convention to compare against good_bitstrings.
def to_le(bitstr):
    return bitstr[::-1]

counts_le = {to_le(k): v for k, v in counts.items()}
sorted_outcomes = sorted(counts_le.items(), key=lambda kv: -kv[1])

print("Top measured outcomes (q0 q1 q2 q3 q4 q5 : shots):")
for bitstr, shots in sorted_outcomes[:8]:
    print(f"  {bitstr} : {shots}")

# --- Step 4: PASS/FAIL check -------------------------------------------
top_k = len(good_bitstrings)
top_outcomes = set(b for b, _ in sorted_outcomes[:top_k])
good_set = set(good_bitstrings)

good_shots = sum(shots for b, shots in counts_le.items() if b in good_set)
total_shots = sum(counts_le.values())
good_fraction = good_shots / total_shots

print(f"Fraction of shots landing on a good (max-edge, C4-free) state: {good_fraction:.3f}")
print(f"Classical answer: max edges = {CLASSICAL_ANSWER['max_edges']}, "
      f"{CLASSICAL_ANSWER['num_optimal_graphs']} optimal graphs")

verified = (top_outcomes == good_set) and (good_fraction > 0.5)

if verified:
    print("PASS")
else:
    print("FAIL")
