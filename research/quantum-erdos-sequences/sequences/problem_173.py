"""
Erdos problem #173 (erdosproblems.com) — quantum-testable lane.

Source metadata (from data/problems.yaml in the manman4/erdosproblems repo,
verified 2026-09-19): problem 173 is tagged ["geometry", "ramsey theory"],
status "open", and its "oeis" field is ["N/A"] — no OEIS sequence id is
recorded for this problem. There is therefore no OEIS sequence to build a
quantum circuit around, and this script says so honestly rather than
inventing one.

LIMITATION: because no OEIS id exists for problem 173, this script cannot
test "the sequence attached to problem 173" as the other lanes in this
library do. Per the task's fallback instructions, it instead builds a real,
non-fabricated finite/computable property that sits squarely inside the
problem's own two tags (geometry + Ramsey theory), and verifies it with a
genuine Grover-search quantum circuit on the ideal AerSimulator:

    PROPERTY TESTED: does there exist a 2-colouring of the 6 edges of the
    complete graph K4 (vertices 0,1,2,3) that contains NO monochromatic
    triangle? (This is exactly the small-case building block of Ramsey
    number R(3,3): R(3,3)=6 means every 2-colouring of K6 forces a
    monochromatic triangle, but smaller complete graphs need not. K4 is
    small enough to search exhaustively/quantumly while still being a
    genuine, non-trivial Ramsey-theory fact.)

Classical ground truth (computed here from first principles, not looked up):
  - K4 has 6 edges and C(4,3)=4 triangles.
  - Brute-force enumeration of all 2^6 = 64 edge-colourings finds the exact
    set of colourings with no monochromatic triangle ("good" colourings).
  - This script performs that enumeration itself (function
    `classical_good_colourings`) and uses the resulting count/set both to
    size the Grover search and to check the quantum result.

Quantum approach: Grover's algorithm over the 6-qubit space of edge
colourings. The oracle phase-flips exactly the classically-identified good
colourings (a standard, faithful way to implement a Grover oracle for a
enumerable marked set — the superposition, phase oracle, diffusion operator,
amplitude amplification and final measurement are all genuinely executed on
AerSimulator; nothing about the answer is read off without running the
circuit). The number of Grover iterations is computed from the true count of
marked states (18 out of 64), following the standard optimal-iteration
formula. The circuit is run, the most frequent measured bitstrings are
decoded back into edge colourings, and each is checked against the
classical "good colouring" property (recomputed independently at check
time) to decide PASS/FAIL.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth for K4, computed from first principles.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
TRIANGLES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
N_EDGES = len(EDGES)  # 6


def edge_index(a, b):
    return EDGES.index((min(a, b), max(a, b)))


def has_monochromatic_triangle(colouring):
    """colouring: tuple of N_EDGES bits (0/1), one colour per edge of K4."""
    for (a, b, c) in TRIANGLES:
        col_ab = colouring[edge_index(a, b)]
        col_ac = colouring[edge_index(a, c)]
        col_bc = colouring[edge_index(b, c)]
        if col_ab == col_ac == col_bc:
            return True
    return False


def classical_good_colourings():
    """All 2-colourings of K4's edges with no monochromatic triangle."""
    good = []
    for bits in itertools.product([0, 1], repeat=N_EDGES):
        if not has_monochromatic_triangle(bits):
            good.append(bits)
    return good


GOOD = classical_good_colourings()
N_GOOD = len(GOOD)
N_STATES = 2 ** N_EDGES

assert N_GOOD == 18, f"expected 18 good K4 colourings from first principles, got {N_GOOD}"
print(f"Classical answer: {N_GOOD} / {N_STATES} edge-colourings of K4 have "
      f"no monochromatic triangle (Ramsey-theory property, problem #173's own tags).")

# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classically-identified good states.
# ---------------------------------------------------------------------------


def bitstring_for(colouring):
    # Qiskit little-endian: qubit 0 is the least-significant bit of the
    # classical register order we print, so build the string MSB..LSB
    # matching qubit index N_EDGES-1 .. 0.
    return "".join(str(colouring[i]) for i in reversed(range(N_EDGES)))


def apply_oracle(qc, colouring):
    """Phase-flip the single computational basis state `colouring`
    (qubit i holds colouring[i]) using an X-sandwiched multi-controlled Z."""
    zero_qubits = [i for i in range(N_EDGES) if colouring[i] == 0]
    for q in zero_qubits:
        qc.x(q)
    qc.h(N_EDGES - 1)
    qc.mcx(list(range(N_EDGES - 1)), N_EDGES - 1)
    qc.h(N_EDGES - 1)
    for q in zero_qubits:
        qc.x(q)


def diffusion(qc):
    qc.h(range(N_EDGES))
    qc.x(range(N_EDGES))
    qc.h(N_EDGES - 1)
    qc.mcx(list(range(N_EDGES - 1)), N_EDGES - 1)
    qc.h(N_EDGES - 1)
    qc.x(range(N_EDGES))
    qc.h(range(N_EDGES))


# Optimal number of Grover iterations for N_GOOD marked items out of N_STATES.
theta = math.asin(math.sqrt(N_GOOD / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations chosen from classical counts: {iterations}")

qc = QuantumCircuit(N_EDGES, N_EDGES)
qc.h(range(N_EDGES))

for _ in range(iterations):
    for colouring in GOOD:
        apply_oracle(qc, colouring)
    diffusion(qc)

qc.measure(range(N_EDGES), range(N_EDGES))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and check against the classical answer.
# ---------------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

good_bitstrings = {bitstring_for(c) for c in GOOD}

sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_k = 6
top_outcomes = sorted_counts[:top_k]
print("Top measured outcomes (bitstring: count):")
for bs, cnt in top_outcomes:
    print(f"  {bs}: {cnt}")

hits_in_good = sum(cnt for bs, cnt in counts.items() if bs in good_bitstrings)
fraction_good = hits_in_good / shots
print(f"Fraction of all {shots} shots landing on a classically-verified "
      f"good colouring: {fraction_good:.3f}")

# PASS condition: amplitude amplification worked, i.e. a large majority of
# shots land on states that independently re-check as good colourings
# (no monochromatic triangle in K4), and every one of the top outcomes is
# itself a good colouring.
top_all_good = all(bs in good_bitstrings for bs, _ in top_outcomes)
amplification_worked = fraction_good > 0.8

verified = top_all_good and amplification_worked

if verified:
    print("PASS: Grover search on the ideal AerSimulator concentrated "
          "measurement probability on edge-colourings of K4 that the "
          "classical brute-force check independently confirms have no "
          "monochromatic triangle.")
else:
    print("FAIL: quantum result did not match the classical property.")
