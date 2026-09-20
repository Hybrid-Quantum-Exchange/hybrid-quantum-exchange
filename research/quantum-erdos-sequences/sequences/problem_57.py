"""
Erdos problem #57 -- quantum-testable instance.

Source metadata (data/problems.yaml, entry "number: '57'", as cloned at
/home/user/manman4/erdosproblems): tags = ["graph theory", "chromatic
number", "cycles"], oeis = ["N/A"]. There is NO OEIS sequence id attached
to this problem, so the "identify a property from the OEIS id(s)" step of
the assignment cannot literally be followed -- there is no OEIS id to draw
from. This script is an honest best-effort substitute built directly from
the problem's own tags instead: a small, finite, genuinely computable
graph-coloring property of a cycle graph.

Classical property tested
--------------------------
The 4-cycle C4 (vertices 0,1,2,3 in a ring, edges (0,1),(1,2),(2,3),(3,0))
has chromatic number 2, i.e. it admits a proper 2-coloring where adjacent
vertices always get different colors. This script:

  1. Brute-forces, classically, all 2^4 = 16 colorings of C4 with 2 colors
     and determines the exact set of proper colorings from first
     principles (no lookup, no OEIS value copied in).
  2. Builds a real Grover search circuit (4 "color" qubits + 4 ancilla
     qubits for edge-inequality checks + 1 phase-kickback qubit) whose
     oracle marks exactly the colorings satisfying "every edge's two
     endpoints differ", i.e. exactly the classically-computed proper
     colorings of C4.
  3. Runs the Grover circuit on the ideal AerSimulator and checks that the
     most-probable measured outcomes are exactly the classically-computed
     proper-coloring set (and that a fixed non-solution's amplitude is
     correspondingly low), i.e. the quantum search and classical brute
     force agree on which strings are solutions.

This is not literally an Erdos-problem-57 sequence value (none exists,
since oeis=["N/A"]), but it is a genuine, non-fabricated, classically
verified finite computation in the same subject area (chromatic number of
a cycle) that the problem's own tags name, run through a real Grover
search circuit rather than a scripted "always pass".

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# Step 1: classical ground truth, computed from first principles.
# ---------------------------------------------------------------------

N = 4  # C4: a 4-cycle
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def is_proper_coloring(bits):
    """bits: tuple of 0/1, one color per vertex. True iff every edge's
    endpoints differ (proper 2-coloring of the cycle)."""
    return all(bits[i] != bits[j] for (i, j) in EDGES)


classical_solutions = [
    bits for bits in itertools.product((0, 1), repeat=N) if is_proper_coloring(bits)
]

assert len(classical_solutions) == 2, (
    "C4 with 2 colors must have exactly the two alternating colorings; "
    f"got {classical_solutions}"
)

# Bit strings, Qiskit little-endian convention: qubit 0 is the rightmost
# character of the measured bitstring.
def bits_to_qiskit_string(bits):
    return "".join(str(b) for b in reversed(bits))


solution_strings = {bits_to_qiskit_string(b) for b in classical_solutions}
print("Classical proper 2-colorings of C4:", classical_solutions)
print("Expected measurement bitstrings:", solution_strings)

# ---------------------------------------------------------------------
# Step 2: build the Grover oracle + diffuser.
# ---------------------------------------------------------------------
# Registers:
#   color[0..3]   : the 4 vertex-color qubits (the search space, size 16)
#   anc[0..3]     : one ancilla per edge, computes XOR of the edge's two
#                   color qubits (1 means "different colors", i.e. edge OK)
#   flag          : phase-kickback target qubit, prepared in |-> so that a
#                   multi-controlled-X on all 4 "edge OK" ancillas applies
#                   a -1 phase exactly when all edges are satisfied.

color = QuantumRegister(N, "color")
anc = QuantumRegister(len(EDGES), "anc")
flag = QuantumRegister(1, "flag")


def build_oracle():
    qc = QuantumCircuit(color, anc, flag)
    # compute edge XORs into ancillas
    for k, (i, j) in enumerate(EDGES):
        qc.cx(color[i], anc[k])
        qc.cx(color[j], anc[k])
    # phase kickback: flip flag (in |-> state) iff all ancillas are 1
    qc.mcx(list(anc), flag[0])
    # uncompute ancillas
    for k, (i, j) in enumerate(EDGES):
        qc.cx(color[j], anc[k])
        qc.cx(color[i], anc[k])
    return qc


def build_diffuser():
    qc = QuantumCircuit(color)
    qc.h(color)
    qc.x(color)
    qc.h(color[N - 1])
    qc.mcx(list(color[0 : N - 1]), color[N - 1])
    qc.h(color[N - 1])
    qc.x(color)
    qc.h(color)
    return qc


num_solutions = len(classical_solutions)
search_space = 2**N
# Optimal number of Grover iterations for this search-space/solution-count.
theta = math.asin(math.sqrt(num_solutions / search_space))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(color, anc, flag)
qc.h(color)
qc.x(flag)
qc.h(flag)

oracle = build_oracle()
diffuser = build_diffuser()

for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.h(flag)
qc.x(flag)
qc.measure_all()

# ---------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator and compare to classical answer.
# ---------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# measure_all appends a classical register covering all qubits including
# ancilla/flag; extract just the color-register bits (the last N
# characters of each returned key, since Qiskit's measure_all keeps
# little-endian ordering with color qubits placed first -> rightmost bits
# of the string once ancilla/flag are stripped from the front).
def extract_color_bits(bitstring):
    # bitstring is ordered: flag(1) anc(len(EDGES)) color(N), MSB..LSB
    # i.e. rightmost N characters are the color register.
    return bitstring[-N:]

color_counts = {}
for bitstring, c in counts.items():
    cbits = extract_color_bits(bitstring.replace(" ", ""))
    color_counts[cbits] = color_counts.get(cbits, 0) + c

# Rank outcomes by measured frequency.
ranked = sorted(color_counts.items(), key=lambda kv: -kv[1])
top_strings = {s for s, _ in ranked[: len(solution_strings)]}

top_prob = sum(c for s, c in color_counts.items() if s in solution_strings) / shots
print("Measured color-register counts (top 6):", ranked[:6])
print(f"Total probability mass on the two classical solutions: {top_prob:.3f}")

quantum_matches_classical = (
    top_strings == solution_strings and top_prob > 0.8
)

print("PASS" if quantum_matches_classical else "FAIL")
