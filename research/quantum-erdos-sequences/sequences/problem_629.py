"""
Erdos problem #629 (from https://github.com/manman4/erdosproblems, data/problems.yaml)
=======================================================================================

Metadata as recorded in the source repository (problems.yaml, entry "number: 629"):
    prize: no
    status: open (informal_status.state = "open", last_update 2025-08-31)
    oeis: ["possible"]
    tags: ["graph theory", "chromatic number"]

LIMITATION, stated up front and honestly:
The "oeis" field for problem #629 is the literal string "possible", not an
actual OEIS sequence id (e.g. not "A000027" or similar). There is therefore
no real OEIS sequence attached to this problem to build a "quantum-testable
sequence membership" test out of -- fabricating an OEIS id here would be
dishonest, and the task instructions explicitly forbid that. The problem's
only other structured content is the tag pair ("graph theory",
"chromatic number"), with no formula, statement text, or numeric data in the
YAML entry to derive a specific finite property from.

Given that, this script does NOT claim to test "the sequence for problem
629" (no such OEIS sequence exists in the source data). Instead, in the
spirit of the problem's tags, it builds a genuine, small, finite,
classically-checkable graph-coloring decision instance -- the kind of
combinatorial object "chromatic number" problems are about -- and solves it
with a real Grover search circuit run on Qiskit's AerSimulator, then checks
the quantum result against a from-scratch classical brute-force computation.

Chosen finite instance
-----------------------
Graph: path graph P3, vertices {0,1,2}, edges {(0,1),(1,2)}.
Property under test: does a proper 2-coloring of P3 exist, and if so, which
3-bit assignments (one bit per vertex, bit = color in {0,1}) are valid?

Classical answer (computed here by brute force over all 2^3 = 8
assignments, first principles, no lookup):
    An assignment (c0,c1,c2) is valid iff c0 != c1 and c1 != c2.
    -> valid assignments: (0,1,0) and (1,0,1), i.e. bit patterns "010" and
       "101" (qubit0=c0 as the least significant bit in circuit order).

Quantum method
---------------
A 3-qubit Grover search circuit is built:
  - Phase oracle flips the sign of exactly the two valid basis states
    |010> and |101> (verified against the classical brute-force set below
    before being hard-coded into the circuit, so the oracle is provably
    correct for this instance rather than assumed).
  - Standard Grover diffuser (inversion about the mean) amplifies those
    two marked states.
  - With N=8 states and M=2 marked, the optimal number of Grover
    iterations is round(pi/4 * sqrt(N/M)) = 1.
  - The circuit is run on qiskit_aer's AerSimulator (ideal, no noise),
    sampled with many shots.

PASS/FAIL
---------
The script computes, from the search space itself (not from any external
source), the set of valid colorings, builds the oracle from that
classically-verified set, runs the Grover circuit, and declares PASS iff
the two most frequent measured outcomes are exactly the two classically
valid colorings ("010" and "101").

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (from first principles): brute-force the proper
#    2-colorings of the path graph P3 with edges (0,1) and (1,2).
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2]
EDGES = [(0, 1), (1, 2)]


def is_proper_coloring(coloring, edges):
    return all(coloring[u] != coloring[v] for (u, v) in edges)


classical_valid = []
for bits in itertools.product([0, 1], repeat=3):
    coloring = {v: bits[v] for v in VERTICES}
    if is_proper_coloring(coloring, EDGES):
        classical_valid.append(bits)

# Represent each valid assignment as the bitstring qiskit will report, with
# qubit i = vertex i and qiskit's little-endian convention (qubit 0 is the
# rightmost character of the classical register string).
classical_valid_bitstrings = set(
    "".join(str(bits[v]) for v in reversed(VERTICES)) for bits in classical_valid
)

print("Classical brute-force search space: 2^3 = 8 colorings")
print("Classically valid proper 2-colorings of P3:", sorted(classical_valid_bitstrings))
assert classical_valid_bitstrings == {"010", "101"}, (
    "Sanity check failed: expected exactly {'010','101'} for path graph P3 "
    f"2-coloring, got {classical_valid_bitstrings}"
)


# ---------------------------------------------------------------------------
# 2. Build a 3-qubit Grover circuit whose oracle marks exactly the
#    classically-valid bitstrings computed above.
# ---------------------------------------------------------------------------

N_QUBITS = 3
N_STATES = 2 ** N_QUBITS
MARKED = classical_valid_bitstrings  # {"010", "101"}, derived above, not assumed


def apply_oracle(qc: QuantumCircuit, marked_bitstrings, qubits):
    """Flip the phase of each basis state in marked_bitstrings.

    Each bitstring is qiskit-ordered (qubit 0 = rightmost char). For a target
    bitstring, X-gate the qubits that should be 0, apply a multi-controlled
    Z (via H + MCX + H on the last qubit), then undo the X-gates.
    """
    n = len(qubits)
    for bitstring in marked_bitstrings:
        # bitstring[k] corresponds to qubits[n-1-k] under little-endian order
        zero_positions = [
            qubits[n - 1 - k] for k, b in enumerate(bitstring) if b == "0"
        ]
        for q in zero_positions:
            qc.x(q)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q in zero_positions:
            qc.x(q)


def apply_diffuser(qc: QuantumCircuit, qubits):
    n = len(qubits)
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


M = len(MARKED)
iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N_STATES / M)))
print(f"Grover iterations used: {iterations} (N={N_STATES}, M={M})")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))
qc.h(qubits)
for _ in range(iterations):
    apply_oracle(qc, MARKED, qubits)
    apply_diffuser(qc, qubits)
qc.measure(qubits, qubits)

sim = AerSimulator()
compiled = transpile(qc, sim)
result = sim.run(compiled, shots=4096).result()
counts = result.get_counts()

print("Measurement counts:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

top2 = set(sorted(counts, key=lambda k: -counts[k])[:2])

verified = top2 == MARKED
if verified:
    print("PASS: Grover search's top measured outcomes match the classically "
          f"computed valid 2-colorings of P3: {sorted(MARKED)}")
else:
    print("FAIL: Grover search's top measured outcomes", sorted(top2),
          "do not match the classical answer", sorted(MARKED))
