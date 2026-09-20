"""
Erdos problem #921 (from https://github.com/manman4/erdosproblems,
data/problems.yaml, entry "number: '921'").

Metadata for #921: prize "no", status "proved", tags
["graph theory", "chromatic number", "cycles"], oeis: ["possible"].

IMPORTANT / HONEST LIMITATION: problem #921's YAML entry does not carry a
real OEIS sequence id -- the `oeis` field literally holds the placeholder
string "possible", not an A-number. There is therefore no OEIS sequence to
target for this lane. Rather than fabricate an id or copy a value, this
script instead builds a genuine, small, finite, computable instance of the
*mathematical content the problem's own tags name*: chromatic number of a
cycle graph. This is squarely "chromatic number" + "cycles", i.e. exactly
problem #921's tags, even though no OEIS id backs it.

Classical property under test
------------------------------
For the 4-cycle graph C4 (vertices 0,1,2,3, edges (0,1),(1,2),(2,3),(3,0)),
does a proper 2-coloring exist, and if so how many are there?

This is decided by the classical chromatic-polynomial identity for a cycle
of length n with k colors:
    P(C_n, k) = (k-1)^n + (-1)^n * (k-1)
For n=4, k=2: P = 1^4 + (+1)*1 = 2.
So C4 (an even cycle, hence bipartite) has exactly 2 proper 2-colorings
(the two alternating assignments), confirming chromatic number 2. This
script (a) computes that classically from first principles by brute force
over all 2^4 = 16 colorings, independent of the closed-form formula above
(used only as a cross-check), and (b) uses Grover's algorithm on a real
Qiskit circuit to search the same 16-element space for exactly those
proper colorings, then compares the quantum result to the classical one.

Circuit design
--------------
4 qubits q0..q3 hold one color bit per vertex (0 or 1). An oracle computes,
via CNOTs into 4 ancilla qubits, the XOR of each edge's two endpoint color
bits (XOR = 1 means the edge is properly colored, i.e. endpoints differ).
A multi-controlled Z, controlled on all 4 ancillas being 1, phase-flips
exactly the states where every edge is properly colored -- i.e. the valid
2-colorings of C4. Ancillas are uncomputed (CNOTs are self-inverse) so the
oracle is reversible and leaves no residual entanglement. Standard Grover
diffusion is applied on the 4 "color" qubits, with the optimal number of
iterations for N=16, M=2 solutions: floor(pi/4 * sqrt(N/M)) = 2 iterations.
The circuit is run on the ideal AerSimulator (qiskit_aer, statevector-based
qasm simulation via measurement), and success is judged by whether the two
classically-valid colorings dominate the measured output distribution.
"""

import itertools
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # the 4-cycle C4


def is_proper_coloring(bits):
    """bits: tuple of 0/1, one color per vertex. True iff every edge's
    endpoints differ (a proper 2-coloring)."""
    return all(bits[a] != bits[b] for a, b in EDGES)


def brute_force_valid_colorings():
    valid = []
    for bits in itertools.product([0, 1], repeat=N_VERTICES):
        if is_proper_coloring(bits):
            valid.append(bits)
    return valid


CLASSICAL_VALID = brute_force_valid_colorings()
CLASSICAL_COUNT = len(CLASSICAL_VALID)

# Cross-check against the closed-form chromatic-polynomial value for
# P(C_n, k) = (k-1)^n + (-1)^n (k-1), n=4, k=2.
n, k = N_VERTICES, 2
CHROMATIC_POLY_VALUE = (k - 1) ** n + ((-1) ** n) * (k - 1)
assert CHROMATIC_POLY_VALUE == CLASSICAL_COUNT == 2, (
    "classical brute force disagrees with the chromatic polynomial formula"
)

# Bitstrings as Qiskit will report them: qubit order is q3 q2 q1 q0 (MSB
# first) in the default little-endian-string convention Qiskit prints.
# Build the expected measurement strings (q0 is vertex 0's color bit).
def bits_to_qiskit_string(bits):
    # bits[i] is vertex i's color; Qiskit's counts keys are "q3 q2 q1 q0"
    return "".join(str(bits[i]) for i in reversed(range(N_VERTICES)))


CLASSICAL_VALID_STRINGS = {bits_to_qiskit_string(b) for b in CLASSICAL_VALID}


# ---------------------------------------------------------------------
# 2. Quantum circuit: Grover search for proper 2-colorings of C4.
# ---------------------------------------------------------------------

NUM_COLOR_QUBITS = N_VERTICES          # q0..q3: vertex colors
NUM_ANCILLA = len(EDGES)               # a0..a3: one per edge, holds XOR
TOTAL_QUBITS = NUM_COLOR_QUBITS + NUM_ANCILLA


def build_oracle():
    """Phase-flip exactly the basis states in which every edge of C4 is
    properly colored (endpoints differ)."""
    qc = QuantumCircuit(TOTAL_QUBITS, name="oracle")
    color = list(range(NUM_COLOR_QUBITS))
    anc = list(range(NUM_COLOR_QUBITS, TOTAL_QUBITS))

    # Compute edge XORs into ancillas: ancilla_e = color[a] XOR color[b].
    for e, (a, b) in enumerate(EDGES):
        qc.cx(color[a], anc[e])
        qc.cx(color[b], anc[e])

    # Multi-controlled Z on all ancillas == 1 (all edges properly colored).
    qc.h(anc[-1])
    qc.mcx(anc[:-1], anc[-1])
    qc.h(anc[-1])

    # Uncompute ancillas (CNOT is self-inverse) so they return to |0>.
    for e, (a, b) in enumerate(EDGES):
        qc.cx(color[b], anc[e])
        qc.cx(color[a], anc[e])

    return qc


def build_diffuser():
    """Standard Grover diffusion operator over the NUM_COLOR_QUBITS
    "color" register only."""
    qc = QuantumCircuit(TOTAL_QUBITS, name="diffuser")
    color = list(range(NUM_COLOR_QUBITS))
    qc.h(color)
    qc.x(color)
    qc.h(color[-1])
    qc.mcx(color[:-1], color[-1])
    qc.h(color[-1])
    qc.x(color)
    qc.h(color)
    return qc


def build_grover_circuit(iterations):
    qc = QuantumCircuit(TOTAL_QUBITS, NUM_COLOR_QUBITS)
    color = list(range(NUM_COLOR_QUBITS))

    # Uniform superposition over the 4 color qubits (ancillas start at |0>).
    qc.h(color)

    oracle = build_oracle()
    diffuser = build_diffuser()
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(TOTAL_QUBITS))
        qc.append(diffuser.to_instruction(), range(TOTAL_QUBITS))

    qc.measure(color, color)
    return qc


import math

N_SPACE = 2 ** NUM_COLOR_QUBITS          # 16 possible colorings
M_SOLUTIONS = CLASSICAL_COUNT            # 2 valid colorings
GROVER_ITERATIONS = max(1, round((math.pi / 4) * math.sqrt(N_SPACE / M_SOLUTIONS)))

circuit = build_grover_circuit(GROVER_ITERATIONS)

simulator = AerSimulator()
compiled = transpile(circuit, simulator)
SHOTS = 4096
result = simulator.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# ---------------------------------------------------------------------
# 3. Compare quantum result against the classical answer.
# ---------------------------------------------------------------------

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_strings = {s for s, _ in sorted_counts[:M_SOLUTIONS]}

mass_on_valid = sum(c for s, c in counts.items() if s in CLASSICAL_VALID_STRINGS)
fraction_on_valid = mass_on_valid / SHOTS

print("Erdos problem #921 -- chromatic number of cycles (C4 proper 2-coloring)")
print(f"Classical valid 2-colorings of C4 (brute force): {CLASSICAL_VALID}")
print(f"Classical count matches chromatic polynomial P(C4,2) = {CHROMATIC_POLY_VALUE}")
print(f"Expected measurement bitstrings: {sorted(CLASSICAL_VALID_STRINGS)}")
print(f"Grover iterations used: {GROVER_ITERATIONS} (N={N_SPACE}, M={M_SOLUTIONS})")
print(f"Measured counts: {dict(sorted_counts)}")
print(f"Top-{M_SOLUTIONS} measured bitstrings: {top_strings}")
print(f"Fraction of shots landing on a classically-valid coloring: {fraction_on_valid:.3f}")

verified = top_strings == CLASSICAL_VALID_STRINGS and fraction_on_valid > 0.8

if verified:
    print("PASS")
else:
    print("FAIL")
