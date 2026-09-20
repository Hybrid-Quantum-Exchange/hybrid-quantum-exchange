"""
Erdos problem #627 (erdosproblems.com / manman4/erdosproblems data/problems.yaml).

Source-data honesty note (read before trusting "verified_against_classical"):
The metadata block for problem 627 in data/problems.yaml is:

    number: "627"
    oeis: ["possible"]
    tags: ["graph theory", "chromatic number"]

"possible" is a placeholder the dataset uses when no real OEIS sequence has
been linked to the problem yet -- it is NOT an OEIS id, and there is no
concrete integer sequence attached to problem 627 to test membership/terms
of. So this script does NOT (and cannot honestly) verify an OEIS term for
problem 627. Per the task's own fallback instruction, this is the "best
honest attempt": it builds a REAL, genuine small quantum circuit around the
one piece of real mathematical content problem 627's metadata does give us
-- its tags, "graph theory" / "chromatic number" -- rather than fabricating
or copying a fake OEIS value.

Classical property being tested (finite, computable, chosen ourselves):
    Is the 4-cycle graph C4 (vertices 0,1,2,3; edges (0,1),(1,2),(2,3),(3,0))
    2-colorable, and if so, which of the 2^4 = 16 possible 2-colorings of its
    vertices are proper (no edge monochromatic)?

This is computed classically in this script by brute force over all 16
colorings (ground truth: exactly 2 proper 2-colorings exist, the two
alternating colorings 0101 and 1010 -- consistent with C4's known chromatic
number, 2, since it is bipartite/even-cycle).

Quantum method: a genuine Grover search circuit.
    - 4 "main" qubits encode a candidate 2-coloring of the 4 vertices.
    - 4 "edge" ancilla qubits compute, via CNOT-XOR, whether each of the 4
      edges is properly colored (endpoints differ).
    - 1 "out" qubit, prepared in the |-> state, is flipped by a 4-controlled-X
      gate on the edge ancillas, which (via phase kickback) marks with a -1
      phase exactly the main-register states that are proper colorings.
    - The edge ancillas are uncomputed (reversed) after each oracle call so
      only the main register stays entangled with the marked phase.
    - A standard Grover diffuser (H / X / multi-controlled-Z / X / H) on the
      4 main qubits amplifies the marked (proper-coloring) states.
    - With N=16 states and M=2 solutions, floor(pi/4 * sqrt(N/M)) = 2 Grover
      iterations is used.

This is run on the ideal AerSimulator (statevector-based sampling), and the
script prints PASS if the two most probable 4-bit outcomes on the main
register match the classical brute-force solution set {0101, 1010} (bit
order q0 q1 q2 q3) with high combined probability, FAIL otherwise.

OEIS id(s) used: NONE -- problem 627's dataset entry has no real OEIS id
(only the placeholder "possible"), so no OEIS sequence membership/term claim
is made anywhere in this script.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute force all 2-colorings of C4.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # 4-cycle


def is_proper_coloring(bits):
    """bits: tuple of 4 ints (0/1), one color per vertex 0..3."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


def classical_solutions():
    sols = []
    for i in range(16):
        bits = tuple((i >> k) & 1 for k in range(4))  # bits[k] = qubit k's value
        if is_proper_coloring(bits):
            sols.append(bits)
    return sols


CLASSICAL_SOLUTIONS = classical_solutions()
assert len(CLASSICAL_SOLUTIONS) == 2, (
    f"expected exactly 2 proper 2-colorings of C4, got {CLASSICAL_SOLUTIONS}"
)
# Ground truth: {(1,0,1,0), (0,1,0,1)} i.e. bitstrings "0101" and "1010"
# (q3 q2 q1 q0 order, matching Qiskit's little-endian classical register printing).
CLASSICAL_BITSTRINGS = set()
for bits in CLASSICAL_SOLUTIONS:
    # Qiskit prints classical registers with qubit 0 as the rightmost bit.
    s = "".join(str(bits[k]) for k in reversed(range(4)))
    CLASSICAL_BITSTRINGS.add(s)

print("Classical brute force over all 16 colorings of C4:")
print(f"  proper 2-colorings found: {sorted(CLASSICAL_BITSTRINGS)}")
print(f"  count = {len(CLASSICAL_BITSTRINGS)} (chromatic number of C4 is 2, as expected)")


# ---------------------------------------------------------------------------
# 2. Quantum Grover search circuit.
# ---------------------------------------------------------------------------

main = QuantumRegister(4, "v")     # vertex color qubits v0..v3
anc = QuantumRegister(4, "e")      # edge-parity ancillas e0..e3
out = QuantumRegister(1, "out")    # phase-kickback output qubit
qc = QuantumCircuit(main, anc, out)

# Initial uniform superposition over all 16 colorings.
qc.h(main)

# Output qubit prepared in |-> for phase-kickback oracle marking.
qc.x(out)
qc.h(out)


def apply_oracle(qc):
    # Compute edge parities into ancillas: e_i = v_i XOR v_{(i+1) mod 4}
    for i, (u, w) in enumerate(EDGES):
        qc.cx(main[u], anc[i])
        qc.cx(main[w], anc[i])
    # Flip out iff ALL 4 edges are properly colored (all ancillas == 1).
    qc.mcx([anc[0], anc[1], anc[2], anc[3]], out[0])
    # Uncompute ancillas.
    for i, (u, w) in enumerate(EDGES):
        qc.cx(main[w], anc[i])
        qc.cx(main[u], anc[i])


def apply_diffuser(qc):
    qc.h(main)
    qc.x(main)
    qc.h(main[3])
    qc.mcx([main[0], main[1], main[2]], main[3])
    qc.h(main[3])
    qc.x(main)
    qc.h(main)


N = 16
M = len(CLASSICAL_BITSTRINGS)
iterations = max(1, round(np.pi / 4 * np.sqrt(N / M)))
print(f"\nRunning Grover search: N={N} states, M={M} marked, iterations={iterations}")

for _ in range(iterations):
    apply_oracle(qc)
    apply_diffuser(qc)

# Uncompute the output qubit back toward |0> convention before measuring
# (not strictly required for correctness of main-register statistics, but
# tidy): restore it from |-> to |0>.
qc.h(out)
qc.x(out)

# Measure only the 4 main (vertex-color) qubits.
from qiskit import ClassicalRegister

creg = ClassicalRegister(4, "c")
qc.add_register(creg)
qc.measure(main, creg)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to classical ground truth.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
job = sim.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

print("\nQuantum measurement counts (top 6):")
for bitstring, c in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
    print(f"  {bitstring}: {c} ({c / shots:.3f})")

marked_prob = sum(counts.get(b, 0) for b in CLASSICAL_BITSTRINGS) / shots
top2 = set(b for b, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:2])

print(f"\nTotal probability mass on classical solution set {sorted(CLASSICAL_BITSTRINGS)}: "
      f"{marked_prob:.3f}")
print(f"Top-2 most frequent measured bitstrings: {sorted(top2)}")

verified = (top2 == CLASSICAL_BITSTRINGS) and (marked_prob > 0.7)

if verified:
    print("\nPASS: Grover search's top outcomes match the classical brute-force "
          "proper-2-coloring set for C4, with high amplified probability.")
else:
    print("\nFAIL: quantum result did not match the classical ground truth "
          "with sufficient confidence.")

print(f"\nran_ok=True verified_against_classical={verified}")
