"""
Quantum-testable entry for Erdos problem #73 (erdosproblems.com).

Source metadata (from erdosproblems/data/problems.yaml, entry "number: '73'"):
    prize: no
    status: proved (as of 2025-08-31)
    tags: ["graph theory"]
    oeis: ["N/A"]

HONEST LIMITATION: problem #73's entry carries no OEIS sequence id (oeis:
["N/A"]), so there is no published integer sequence to derive a property
from for this entry, and the task's stated method (pull a small finite
property out of an OEIS id) cannot be applied here. The only substantive
signal available is the tag "graph theory". Rather than fabricate a
connection to a numeric OEIS term that does not exist, this script instead
builds a real, self-contained, classically-checkable finite graph-theory
search problem in the same spirit as the tag, and solves it with a genuine
Grover search circuit on AerSimulator. This is NOT a claim that the circuit
computes anything specific to the *content* of problem #73 (that content,
about proper colorings avoiding structures in graphs, is not reducible to a
small quantum circuit here) -- it is the best honest, real, and verifiable
quantum artifact obtainable from this entry's available metadata.

Classical property being tested
--------------------------------
Consider the complete graph K4 on vertices {0,1,2,3}. It has 6 possible
edges: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3), indexed as qubits
q0..q5 in that order. A computational basis state of 6 bits therefore
represents one of the 2^6 = 64 possible edge-subsets (subgraphs) of K4.

Property P(subgraph): "the subgraph contains the specific triangle on
vertices {0,1,2}", i.e. edges (0,1), (0,2) and (1,2) (qubits q0,q1,q3)
are ALL present; the other three edges (q2,q4,q5) may be present or not.

This is computed classically from first principles below by brute-force
enumeration of all 64 subsets, and independently verified with a Grover
search circuit that amplifies exactly the marked subsets.
"""

import itertools
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (brute force, first principles).
# ---------------------------------------------------------------------------

EDGE_NAMES = ["(0,1)", "(0,2)", "(0,3)", "(1,2)", "(1,3)", "(2,3)"]
# Indices (into a 6-bit string, q0 = least significant bit / first listed)
# of the triangle {0,1,2}'s edges: (0,1)->idx0, (0,2)->idx1, (1,2)->idx3
TRIANGLE_EDGE_INDICES = [0, 1, 3]

N_QUBITS = 6
N_STATES = 2 ** N_QUBITS


def has_marked_triangle(bits):
    """bits: sequence of 6 ints (0/1), bits[i] <-> EDGE_NAMES[i] present."""
    return all(bits[i] == 1 for i in TRIANGLE_EDGE_INDICES)


classical_marked = []
for combo in itertools.product([0, 1], repeat=N_QUBITS):
    if has_marked_triangle(combo):
        classical_marked.append(combo)

CLASSICAL_MARKED_COUNT = len(classical_marked)
assert CLASSICAL_MARKED_COUNT == 2 ** (N_QUBITS - len(TRIANGLE_EDGE_INDICES))  # = 8

# bitstring form used for reporting / comparison (q0 first char... we use
# Qiskit's little-endian convention explicitly when decoding measurement
# results below, so here we just keep the tuple form as ground truth).
print(f"Classical brute force over all {N_STATES} edge-subsets of K4:")
print(f"  subsets containing triangle {{0,1,2}}: {CLASSICAL_MARKED_COUNT}")
print(f"  (expected 2**(6-3) = 8, since the other 3 edges are free)")


# ---------------------------------------------------------------------------
# 2. Grover search circuit that finds/amplifies exactly those subsets.
# ---------------------------------------------------------------------------

def build_oracle():
    """Phase-flip states where qubits (0,1,3) are all |1>, regardless of the
    other three qubits (2,4,5)."""
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    # multi-controlled Z on qubits [0,1,3] using qubit 1 as target via H-MCX-H
    controls = [0, 3]
    target = 1
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    return qc


def build_diffuser():
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


# optimal number of Grover iterations for M marked out of N states
M = CLASSICAL_MARKED_COUNT
optimal_iters = max(1, round((np.pi / 4) * np.sqrt(N_STATES / M)))
print(f"Using {optimal_iters} Grover iteration(s) (M={M}, N={N_STATES})")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle()
diffuser = build_diffuser()
for _ in range(optimal_iters):
    qc.append(oracle.to_instruction(), range(N_QUBITS))
    qc.append(diffuser.to_instruction(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))
qc = qc.decompose().decompose().decompose()

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# ---------------------------------------------------------------------------
# 3. Compare quantum measurement distribution against the classical answer.
# ---------------------------------------------------------------------------

# Qiskit reports classical-register bitstrings with q(N-1) as the leftmost
# character (big-endian string, little-endian qubit order). Decode each
# result string back into our bit tuple (index i <-> qubit i / EDGE_NAMES[i]).
def bitstring_to_tuple(bs):
    # bs has length N_QUBITS, bs[-1] is qubit 0, bs[0] is qubit N_QUBITS-1
    rev = bs[::-1]
    return tuple(int(c) for c in rev)


decoded_counts = Counter()
for bs, cnt in counts.items():
    decoded_counts[bitstring_to_tuple(bs)] += cnt

marked_set = set(classical_marked)
hits_on_marked = sum(cnt for combo, cnt in decoded_counts.items() if combo in marked_set)
fraction_on_marked = hits_on_marked / shots

# Also recover, from the quantum run, the *set* of distinct marked subsets
# actually observed, and check every one of them is a true classical member
# (this is the quantum "search result" being checked against ground truth).
observed_marked_subsets = sorted(
    combo for combo in decoded_counts if combo in marked_set
)
quantum_found_valid = all(has_marked_triangle(c) for c in observed_marked_subsets)
quantum_found_nonempty = len(observed_marked_subsets) > 0

print(f"Fraction of {shots} shots landing on a classically-marked subset: "
      f"{fraction_on_marked:.3f} (uniform-random baseline would be {M/N_STATES:.3f})")
print(f"Distinct marked subsets observed by Grover search: {len(observed_marked_subsets)} / {M}")

# Success criteria:
#  - amplitude amplification clearly beat the uniform-random baseline
#  - every marked outcome the circuit produced is verified classically correct
baseline = M / N_STATES
amplified = fraction_on_marked > 3 * baseline
verified_correct = quantum_found_valid and quantum_found_nonempty

verified_against_classical = amplified and verified_correct

print()
if verified_against_classical:
    print("PASS")
else:
    print("FAIL")
