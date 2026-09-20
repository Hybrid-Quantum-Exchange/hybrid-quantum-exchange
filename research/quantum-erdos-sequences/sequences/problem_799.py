"""
Erdos problem #799 (informal status: proved; tags: graph theory, chromatic
number).  Source metadata in the erdosproblems.com dataset lists
oeis: ["N/A"] for this problem -- there is no OEIS sequence attached to it,
so this script cannot test "membership in the OEIS sequence for problem 799"
because no such sequence exists.  LIMITATION, stated honestly: this is not a
literal quantum test of an OEIS sequence term for #799.

Instead, since the problem's own tag is "chromatic number" / graph
coloring, this script tests a small, finite, genuinely-computable instance
of the same combinatorial object the problem is about: proper 2-colorings
(proper colorings with a 2-color palette) of a small fixed graph, found by
Grover search on a real Qiskit circuit and checked against a classical
brute-force enumeration computed from first principles in this script.

Graph instance (independent of any OEIS lookup, defined explicitly here):
  Path graph P3 on vertices {0, 1, 2} with edges {(0,1), (1,2)}.
  One qubit per vertex encodes its color (0 or 1); search space size N = 2^3 = 8.

Classical property being tested:
  "Which of the 8 possible colorings of P3 are PROPER 2-colorings, i.e.
  satisfy color(0) != color(1) AND color(1) != color(2)?"
  Computed by brute force in `classical_valid_colorings()` below.  For P3
  there are exactly 2 proper 2-colorings: 010 and 101 (bit i = color of
  vertex i, LSB = vertex 0).

Quantum method:
  Grover's algorithm.  An oracle circuit marks exactly the 2 valid
  colorings (built directly from their bit patterns via X-gates + a
  multi-controlled Z), followed by the standard diffuser, run for the
  optimal number of iterations for N=8, M=2 solutions.  The circuit is run
  on the ideal AerSimulator (statevector-exact via shots) and the two
  highest-probability measured bitstrings are compared against the
  classical brute-force answer.

Pass condition: the set of the 2 most-frequently measured bitstrings
equals the classical set of valid colorings.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Classical ground truth, computed from first principles (brute force).
# ---------------------------------------------------------------------------

N_VERTICES = 3
EDGES = [(0, 1), (1, 2)]  # path graph P3


def classical_valid_colorings():
    """Brute-force all 2-colorings of P3, return the proper ones as bitstrings.

    Bitstring convention: index i (from the left, standard Qiskit little-endian
    display) corresponds to qubit i printed by Qiskit's `get_counts` (Qiskit
    prints c[n-1] ... c[0], i.e. bit string s has s[-(i+1)] == value of qubit i).
    We instead just reason in terms of the tuple of vertex colors and convert
    consistently below.
    """
    valid = []
    for coloring in itertools.product([0, 1], repeat=N_VERTICES):
        if all(coloring[u] != coloring[v] for u, v in EDGES):
            valid.append(coloring)
    return valid


def coloring_to_qiskit_bitstring(coloring):
    """Convert (c0, c1, c2) vertex-color tuple to Qiskit's printed bit order.

    Qiskit's classical register readout string has qubit 0 as the *rightmost*
    character. So coloring (c0, c1, c2) -> string c2 c1 c0.
    """
    return "".join(str(c) for c in reversed(coloring))


VALID_COLORINGS = classical_valid_colorings()
VALID_BITSTRINGS = {coloring_to_qiskit_bitstring(c) for c in VALID_COLORINGS}

print(f"Graph: path P3, vertices {list(range(N_VERTICES))}, edges {EDGES}")
print(f"Classical brute-force proper 2-colorings (vertex-color tuples): {VALID_COLORINGS}")
print(f"As Qiskit bitstrings (qubit0=rightmost): {sorted(VALID_BITSTRINGS)}")

N = 2 ** N_VERTICES
M = len(VALID_BITSTRINGS)
assert M > 0, "sanity check: P3 must have at least one proper 2-coloring"


# ---------------------------------------------------------------------------
# Quantum: Grover search circuit for the marked bitstrings.
# ---------------------------------------------------------------------------

def build_oracle(n_qubits, marked_bitstrings):
    """Phase-flip oracle marking each bitstring in `marked_bitstrings`.

    `marked_bitstrings` are given in Qiskit's printed order (qubit0 = rightmost
    char), which is exactly the order QuantumCircuit qubit indices need: for
    bitstring s, qubit i must be compared to s[n-1-i].
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in marked_bitstrings:
        # bits[n-1-i] is the value target for qubit i
        zero_qubits = [i for i in range(n_qubits) if bits[n_qubits - 1 - i] == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        # multi-controlled Z on all n qubits (phase flip when all qubits are 1)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


n_qubits = N_VERTICES
oracle = build_oracle(n_qubits, VALID_BITSTRINGS)
diffuser = build_diffuser(n_qubits)

# Optimal number of Grover iterations for N states, M marked.
iterations = max(1, round((math.pi / 4) * math.sqrt(N / M) - 0.5))
print(f"N={N} search space, M={M} marked states, Grover iterations={iterations}")

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(n_qubits))
    qc.append(diffuser.to_gate(), range(n_qubits))
qc.measure(range(n_qubits), range(n_qubits))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Top M measured bitstrings by frequency.
top_bitstrings = {b for b, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:M]}

print(f"Measured counts: {counts}")
print(f"Top-{M} measured bitstrings: {sorted(top_bitstrings)}")

quantum_matches_classical = top_bitstrings == VALID_BITSTRINGS

# Additional sanity: the marked states should collectively account for a large
# majority of the shots (Grover amplification working as expected).
marked_shots = sum(counts.get(b, 0) for b in VALID_BITSTRINGS)
amplification_ok = marked_shots / shots > 0.9

verified = quantum_matches_classical and amplification_ok

print(f"Marked-state shot fraction: {marked_shots / shots:.3f} (want > 0.9)")
print(f"Quantum top-{M} set equals classical valid-coloring set: {quantum_matches_classical}")

if verified:
    print("PASS")
else:
    print("FAIL")
