"""
Erdos problem #163 (Burr-Erdos conjecture, graph theory / Ramsey theory).

Source metadata (data/problems.yaml, manman4/erdosproblems, read-only clone):
    number: "163"
    tags: ["graph theory", "ramsey theory"]
    oeis: ["N/A"]

LIMITATION, stated up front: problem #163 has no associated OEIS sequence
("N/A" in the source metadata), so there is no OEIS-derived integer sequence
to test membership/growth of on a quantum circuit. To still build a genuine,
non-fabricated, finite/computable instance tied to this problem's own subject
matter (Ramsey theory of graphs, which is exactly what Burr-Erdos is about),
this script tests the following small, classically-checkable property and
uses Grover's algorithm to *find* witnesses of it on the ideal AerSimulator:

    Classical property tested:
        Among the 2-colorings of the 6 edges of the complete graph K4
        (vertices {0,1,2,3}), how many colorings have NO monochromatic
        triangle (i.e. no 3-clique all three of whose edges share a color)?
        This is exactly the finite question behind the classical Ramsey
        number R(3,3): R(3,3)=6 means every 2-coloring of K6's edges forces
        a monochromatic triangle, while K4 (and K5) admit colorings that
        avoid one. This script brute-forces, in Python, the exact count and
        the exact set of "good" (triangle-avoiding) colorings of K4, then
        uses Grover search to recover a good coloring on a quantum circuit,
        and checks the quantum result against the classical enumeration.

    Encoding: 6 qubits, one per edge of K4.
        Edges (vertex pairs), in qubit order q0..q5:
            q0=(0,1) q1=(0,2) q2=(0,3) q3=(1,2) q4=(1,3) q5=(2,3)
        Triangles (each is a set of 3 edges):
            {0,1,2}: q0,q1,q3
            {0,1,3}: q0,q2,q4
            {0,2,3}: q1,q2,q5
            {1,2,3}: q3,q4,q5
        A coloring (bitstring of length 6, 0/1 = two colors) is "good" if no
        triangle's three edge-qubits are all equal (all 0 or all 1).

    The oracle marks (phase-flips) exactly the "good" bitstrings, computed
    classically first, then realized as a Grover oracle over N=2^6=64 basis
    states with M = |good colorings| marked states, using the standard
    number of Grover iterations floor(pi/4 * sqrt(N/M)).

PASS criterion: the bitstring most frequently measured by the quantum
circuit must be one of the classically-computed "good" colorings (i.e. a
real triangle-avoiding 2-coloring of K4), and Grover must amplify the good
subspace to a majority of the measured shots (P(good) > 0.5), versus the
uniform-random baseline M/N which is well below 0.5 for this instance.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ----------------------------------------------------------------------
# 1. Classical ground truth (first principles, computed here, not looked up)
# ----------------------------------------------------------------------

N_EDGES = 6
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

TRIANGLES = []
for tri in itertools.combinations(range(4), 3):
    idxs = []
    for pair in itertools.combinations(sorted(tri), 2):
        idxs.append(EDGE_INDEX[pair])
    TRIANGLES.append(tuple(idxs))  # 4 triangles, 3 edge-indices each


def is_good(bits):
    """bits: tuple of 0/1 of length 6 (q0..q5). True if no monochromatic triangle."""
    for tri in TRIANGLES:
        vals = [bits[i] for i in tri]
        if vals[0] == vals[1] == vals[2]:
            return False
    return True


GOOD = []
for combo in itertools.product([0, 1], repeat=N_EDGES):
    if is_good(combo):
        GOOD.append(combo)

N = 2 ** N_EDGES
M = len(GOOD)

assert M > 0, "classical brute force found no valid coloring - K4 should admit some"
print(f"Classical brute force: N={N} total colorings, M={M} triangle-avoiding colorings")
print(f"Classical baseline probability of hitting a good coloring uniformly at random: {M / N:.4f}")


def bits_to_bitstring(bits):
    # Qiskit's measurement bitstring is ordered c[n-1]...c[0], i.e. q0 is the
    # rightmost character. We build a helper both ways for clarity.
    return "".join(str(b) for b in reversed(bits))  # q5 q4 ... q0 -> matches Qiskit output order


GOOD_BITSTRINGS = {bits_to_bitstring(b) for b in GOOD}


# ----------------------------------------------------------------------
# 2. Grover oracle + diffuser over the 6-qubit edge-coloring space
# ----------------------------------------------------------------------

def apply_multi_controlled_z(qc, qubits):
    """Phase-flip the |11...1> state of `qubits` (multi-controlled Z)."""
    if len(qubits) == 1:
        qc.z(qubits[0])
        return
    target = qubits[-1]
    controls = qubits[:-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)


def oracle_for_bitstring(qc, qubits, bitstring):
    """Phase-flip exactly the computational basis state matching `bitstring`
    (Qiskit-order string, q_{n-1}...q_0)."""
    n = len(qubits)
    # bitstring[0] corresponds to the most-significant qubit qubits[-1] (=q_{n-1})
    # bitstring[i] corresponds to qubits[n-1-i]
    zero_positions = []
    for i, ch in enumerate(bitstring):
        qubit = qubits[n - 1 - i]
        if ch == "0":
            zero_positions.append(qubit)
    for q in zero_positions:
        qc.x(q)
    apply_multi_controlled_z(qc, qubits)
    for q in zero_positions:
        qc.x(q)


def build_oracle(n, marked_bitstrings):
    qc = QuantumCircuit(n, name="oracle")
    qubits = list(range(n))
    for bs in marked_bitstrings:
        oracle_for_bitstring(qc, qubits, bs)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qubits = list(range(n))
    qc.h(qubits)
    qc.x(qubits)
    apply_multi_controlled_z(qc, qubits)
    qc.x(qubits)
    qc.h(qubits)
    return qc


n = N_EDGES
iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations used: {iterations}")

oracle = build_oracle(n, GOOD_BITSTRINGS)
diffuser = build_diffuser(n)

qc = QuantumCircuit(n, n)
qc.h(range(n))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(n))
    qc.append(diffuser.to_gate(), range(n))
qc.measure(range(n), range(n))

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_bitstring, top_count = sorted_counts[0]

good_shots = sum(c for bs, c in counts.items() if bs in GOOD_BITSTRINGS)
p_good = good_shots / shots

print(f"Most frequent measured bitstring: {top_bitstring} (count {top_count}/{shots})")
print(f"Fraction of shots landing on a classically-valid good coloring: {p_good:.4f}")

top_is_good = top_bitstring in GOOD_BITSTRINGS
amplified = p_good > 0.5 and p_good > (M / N)

if top_is_good and amplified:
    print("PASS")
else:
    print("FAIL")
