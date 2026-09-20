"""
Erdos problem #79 (source: erdosproblems.com data, problems.yaml).

Metadata found for problem #79 in the read-only clone at
/home/user/manman4/erdosproblems/data/problems.yaml:

    number: "79"
    status: proved (2025-08-31)
    tags: ["graph theory", "ramsey theory"]
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #79 carries no OEIS sequence id
("N/A"). There is therefore no actual sequence to build a "quantum-testable
sequence" entry around for this problem number. Rather than fabricate an OEIS
id or copy an unrelated one, this script instead builds a genuine, finite,
computable instance of the same combinatorial family the problem's own tags
name (graph theory / Ramsey theory), since that is the only honest content
available from the metadata: Ramsey-type edge-colouring existence, which is
exactly the kind of statement Erdos-flavoured Ramsey-theory problems concern.

Classical property under test
------------------------------
Let K4 be the complete graph on 4 vertices (vertices 0,1,2,3), with its 6
edges indexed 0..5 in the fixed order:

    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

A 2-colouring of the edges is a bitstring of length 6 (bit i = colour of
edges[i], 0 or 1). K4 has 4 triangles: (0,1,2), (0,1,3), (0,2,3), (1,2,3).
A colouring is "triangle-free" (Ramsey-good) if none of these 4 triangles is
monochromatic (all 3 of its edges the same colour).

This is a small, finite, fully computable search problem in the Ramsey-theory
family: does a monochromatic-triangle-free 2-colouring of K4 exist, and which
bitstrings realise it? (Erdos-style Ramsey theory is precisely the study of
guaranteed monochromatic substructures, e.g. R(3,3)=6 meaning K5 can still be
2-coloured triangle-free but K6 cannot; K4 is a smaller, circuit-friendly
instance of the same question.)

The classical answer (brute force over all 2^6 = 64 colourings, computed by
this script from first principles below) is that exactly 18 of the 64
colourings are monochromatic-triangle-free.

Quantum approach
-----------------
Grover's algorithm on 6 qubits (one per edge) searches for triangle-free
colourings. The oracle is built directly (no external oracle-compiler): for
each of the 18 marked bitstrings, we conjugate a multi-controlled Z by X
gates so it flips the phase of exactly that basis state. With N=64 and
M=18 marked states, the optimal number of Grover iterations is
round(pi/4 * sqrt(N/M)) ~= 1. We run the circuit on the ideal AerSimulator,
sample it, and check that the measured bitstring is one of the 18
classically-verified triangle-free colourings (i.e. the quantum search
result is verified against the classical answer).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, brute force over all 2^6 = 64
#    edge-colourings of K4).
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def triangle_edge_indices(tri):
    a, b, c = tri
    return [
        EDGE_INDEX[tuple(sorted((a, b)))],
        EDGE_INDEX[tuple(sorted((b, c)))],
        EDGE_INDEX[tuple(sorted((a, c)))],
    ]


TRIANGLE_EDGES = [triangle_edge_indices(t) for t in TRIANGLES]


def is_triangle_free(bits):
    """bits: tuple of 6 ints (0/1), one per edge in EDGES order."""
    for te in TRIANGLE_EDGES:
        vals = [bits[i] for i in te]
        if vals[0] == vals[1] == vals[2]:
            return False
    return True


def classical_brute_force():
    valid = []
    for bits in itertools.product([0, 1], repeat=6):
        if is_triangle_free(bits):
            valid.append(bits)
    return valid


VALID_COLOURINGS = classical_brute_force()
N = 64          # 2^6 search space
M = len(VALID_COLOURINGS)  # marked states

print(f"Classical brute force: {M} of {N} edge-colourings of K4 are "
      f"monochromatic-triangle-free.")
assert M == 18, f"expected 18 triangle-free colourings, got {M}"

# Bitstring -> qiskit qubit ordering convention: qubit i holds bit i of the
# colouring, and Qiskit reports measurement strings with qubit (n-1) first
# (i.e. bit order is reversed relative to our tuple). We account for this
# when interpreting results below.


def bits_to_label(bits):
    """Convert a (b0,...,b5) tuple (edge order) to a Qiskit classical-bit
    string 'b5 b4 b3 b2 b1 b0' as it would be printed by qiskit (MSB first,
    corresponding to qubit 5 first)."""
    return "".join(str(b) for b in reversed(bits))


VALID_LABELS = {bits_to_label(b) for b in VALID_COLOURINGS}

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle: phase-flip exactly the marked colourings.
# ---------------------------------------------------------------------------

NUM_QUBITS = 6


def apply_oracle(qc):
    for bits in VALID_COLOURINGS:
        # Flip to |1...1> on the qubits that should be 0 for this bitstring.
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # Multi-controlled Z across all NUM_QUBITS qubits (phase flip iff
        # all qubits are |1>): use H on last qubit + multi-controlled X +
        # H, the standard MCZ-from-MCX construction.
        qc.h(NUM_QUBITS - 1)
        qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
        qc.h(NUM_QUBITS - 1)
        for i in zero_positions:
            qc.x(i)


def apply_diffuser(qc):
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))


iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Running Grover search with {iterations} iteration(s) "
      f"(N={N}, M={M}).")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    apply_oracle(qc)
    apply_diffuser(qc)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and verify against the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
SHOTS = 4096
result = backend.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# Fraction of shots landing on a classically-verified triangle-free colouring.
hits = sum(c for label, c in counts.items() if label in VALID_LABELS)
hit_fraction = hits / SHOTS

most_likely_label = max(counts, key=counts.get)
most_likely_is_valid = most_likely_label in VALID_LABELS

print(f"Most frequent measured bitstring: {most_likely_label} "
      f"({counts[most_likely_label]}/{SHOTS} shots)")
print(f"Fraction of shots on a classically-verified triangle-free "
      f"colouring: {hit_fraction:.3f}")

# Success criterion: Grover amplification must concentrate probability on
# marked (valid) states well above the uniform baseline (M/N = 18/64 ~ 0.281),
# and the single most-likely measured outcome must itself be a classically
# verified triangle-free colouring.
baseline = M / N
verified = most_likely_is_valid and hit_fraction > baseline

print(f"Uniform baseline probability of hitting a marked state: "
      f"{baseline:.3f}")

if verified:
    print("PASS")
else:
    print("FAIL")
