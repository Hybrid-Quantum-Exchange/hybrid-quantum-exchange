"""
Erdos problem #63 (from https://github.com/manman4/erdosproblems,
data/problems.yaml, number: "63").

Metadata found for problem 63:
    prize: "no"
    status: "proved (Lean)"
    oeis: ["N/A"]
    tags: ["graph theory", "chromatic number", "cycles"]

LIMITATION, stated up front: problem 63 has no associated OEIS sequence
(oeis: ["N/A"]) in the data file, so this is not literally a "sequence"
lane -- there is no integer sequence to search for a term of. What the
tags do give is a well-defined, finite, computable property from the
same area of mathematics (graph coloring of cycles), and that is what
is implemented and tested here as the best-effort substitute: a real
Grover search circuit that finds proper 2-colorings of the 4-cycle C4
(vertices 0-1-2-3-0), i.e. it searches the space of all 2^4 = 16
assignments of 2 colors to the 4 vertices of C4 for the ones in which
every edge joins two differently-colored vertices.

Classical ground truth (computed in this script, not copied from
anywhere): C4 is bipartite (an even cycle), so it has exactly 2 proper
2-colorings: the two alternating colorings 0101 and 1010 (bit i = color
of vertex i). This is verified below by exhaustive brute force over all
16 assignments before the quantum circuit is ever built.

Quantum approach:
  - 4 "vertex" qubits v0..v3 hold a candidate coloring.
  - 4 ancilla qubits a0..a3 compute the edge-XOR parity for each edge
    (0,1), (1,2), (2,3), (3,0) via CNOTs: edge is "satisfied" (colors
    differ) iff the corresponding ancilla ends up in state |1>.
  - A phase oracle flips the sign of the state iff all 4 ancillas are
    |1>, i.e. iff the candidate coloring is proper. The ancilla
    computation is uncomputed afterwards (it is discarded, unused, and
    always returned to |0>) so the oracle acts purely as a phase flip
    on the vertex register, as required for Grover's algorithm.
  - Standard Grover diffusion operator on the 4 vertex qubits.
  - With N = 16 states and M = 2 marked states, the optimal number of
    Grover iterations is round(pi/4 * sqrt(N/M)) = round(pi/4*sqrt(8)) = 2.

The circuit is run on the ideal AerSimulator (statevector method) with
many shots, and PASS/FAIL is decided by checking that the two
measurement outcomes with the classically-correct highest probability
are exactly the two true proper 2-colorings of C4, and that together
they carry the large majority of the measured probability mass (Grover
amplification actually happened, not a fluke).

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # the 4-cycle C4
N_VERTICES = 4


def is_proper_2_coloring(bits):
    """bits: tuple of 4 ints in {0,1}, bits[i] = color of vertex i."""
    return all(bits[i] != bits[j] for (i, j) in EDGES)


classical_valid = [bits for bits in product([0, 1], repeat=N_VERTICES)
                   if is_proper_2_coloring(bits)]

# Sanity: C4 is bipartite -> exactly 2 proper 2-colorings.
assert len(classical_valid) == 2, f"expected 2, got {classical_valid}"

# bitstrings as written little-endian (qiskit convention: bit0 = v0, ...,
# rightmost printed character = highest-index qubit). We'll compare using
# integers to avoid string-order confusion.
def bits_to_int(bits):
    val = 0
    for i, b in enumerate(bits):
        val |= (b << i)
    return val

classical_valid_ints = sorted(bits_to_int(b) for b in classical_valid)
print("Classical brute force over all 16 colorings of C4:")
print("  proper 2-colorings (vertex bitstrings, bit i = color of vertex i):",
      classical_valid, "-> ints", classical_valid_ints)


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle + diffusion circuit.
# ---------------------------------------------------------------------------

n = N_VERTICES  # 4 vertex qubits: indices 0..3
anc_offset = n  # ancilla qubits: indices 4..7
total_qubits = 2 * n


def build_oracle():
    qc = QuantumCircuit(total_qubits, name="oracle")
    # Compute edge-parity ancillas: a_k = v_i XOR v_j for edge k = (i, j).
    for k, (i, j) in enumerate(EDGES):
        a = anc_offset + k
        qc.cx(i, a)
        qc.cx(j, a)
    # Phase flip iff all 4 ancillas are |1> (multi-controlled Z on ancillas).
    anc_qubits = list(range(anc_offset, anc_offset + len(EDGES)))
    qc.h(anc_qubits[-1])
    qc.mcx(anc_qubits[:-1], anc_qubits[-1])
    qc.h(anc_qubits[-1])
    # Uncompute the ancillas (reverse order, CX is self-inverse).
    for k, (i, j) in reversed(list(enumerate(EDGES))):
        a = anc_offset + k
        qc.cx(j, a)
        qc.cx(i, a)
    return qc


def build_diffusion():
    qc = QuantumCircuit(total_qubits, name="diffusion")
    vs = list(range(n))
    qc.h(vs)
    qc.x(vs)
    qc.h(vs[-1])
    qc.mcx(vs[:-1], vs[-1])
    qc.h(vs[-1])
    qc.x(vs)
    qc.h(vs)
    return qc


def build_grover_circuit(iterations):
    qc = QuantumCircuit(total_qubits, n)
    vs = list(range(n))
    qc.h(vs)  # ancillas start and stay at |0> outside oracle calls
    oracle = build_oracle()
    diffusion = build_diffusion()
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
    qc.measure(vs, list(range(n)))
    return qc


# Optimal iteration count for N=16, M=2 marked states.
N_STATES = 2 ** n
M_MARKED = len(classical_valid_ints)
iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M_MARKED)))
print(f"Grover search: N={N_STATES} states, M={M_MARKED} marked, "
      f"using {iterations} iteration(s).")

circuit = build_grover_circuit(iterations)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator(method="statevector")
compiled = transpile(circuit, sim)
shots = 20000
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit prints classical-register bitstrings MSB-first as v3 v2 v1 v0.
# Convert each measured bitstring back to our vertex-bit convention.
def creg_str_to_int(s):
    # s is e.g. "0101" with s[0] = v3 (MSB) ... s[-1] = v0 (LSB)
    bits = s[::-1]
    return int(bits, 2)

measured = {}
for bitstring, c in counts.items():
    measured[creg_str_to_int(bitstring)] = measured.get(creg_str_to_int(bitstring), 0) + c

sorted_outcomes = sorted(measured.items(), key=lambda kv: -kv[1])
print("Top measured outcomes (vertex-coloring int -> counts):")
for val, c in sorted_outcomes[:6]:
    print(f"  {val:04b} : {c} ({c/shots:.3f})")

top2 = sorted([val for val, _ in sorted_outcomes[:2]])
top2_mass = sum(c for val, c in sorted_outcomes[:2]) / shots


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

matches_classical_set = (top2 == classical_valid_ints)
amplified_enough = top2_mass > 0.85  # Grover should concentrate probability

print(f"\nClassical proper 2-colorings (as ints): {classical_valid_ints}")
print(f"Quantum top-2 measured outcomes (as ints): {top2}")
print(f"Probability mass on top-2 outcomes: {top2_mass:.3f}")

if matches_classical_set and amplified_enough:
    print("PASS")
else:
    print("FAIL")
