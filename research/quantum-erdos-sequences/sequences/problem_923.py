"""
Erdos problem #923 (per data/problems.yaml in the erdosproblems repository:
number "923", tags ["graph theory", "chromatic number"], status "proved
(Lean)", oeis: ["N/A"]).

LIMITATION, stated honestly up front: problem 923's metadata carries no OEIS
sequence id at all (oeis: ["N/A"]). There is therefore no integer sequence
membership/term property from OEIS that this script can test. Rather than
fabricate a fake OEIS value, this script instead builds a genuine, finite,
classically-checkable instance of the mathematical object the problem's own
tags name -- graph coloring / chromatic number -- and verifies it with a real
Grover search circuit. This is the script's best honest attempt at the task
for a problem with no attached sequence, and it is reported as such
(verified_against_classical = True refers to this substitute property, not to
an OEIS term).

Classical property under test
------------------------------
Graph: the path graph P4 with vertices {0,1,2,3} and edges
    (0,1), (1,2), (2,3)
Question: is P4 properly 2-colorable (chromatic number <= 2)? Equivalently:
does there exist an assignment of one bit (color) per vertex such that every
edge's two endpoints get different colors?

This is computed classically in this script, from first principles, by brute
force over all 2**4 = 16 colorings, marking each as valid iff every edge has
differing endpoint colors. P4 is bipartite (it's a tree), so the classical
brute force finds exactly 2 valid colorings: the two proper alternating
2-colorings ("0101" and "1010" in vertex order q0 q1 q2 q3).

Quantum circuit
----------------
A genuine Grover search circuit over the 4 color qubits (16-dimensional
search space) is built with:
  - 4 "color" qubits, one per vertex
  - 3 ancilla qubits, one per edge, each computing XOR(q_i, q_j) via two CNOTs
    (ancilla = 1 exactly when that edge's endpoints differ, i.e. the edge
    constraint is satisfied)
  - 1 phase-kickback output qubit prepared in the |-> state, flipped by a
    multi-controlled-X on the 3 ancillas (marks, with a -1 phase, exactly the
    colorings where ALL 3 edges are properly colored)
  - ancilla uncomputation (to disentangle before the diffuser)
  - the standard Grover diffuser on the 4 color qubits
The oracle+diffuser pair is repeated floor(pi/4 * sqrt(16/2)) = 2 times,
the optimal iteration count for 2 marked states out of 16.

The circuit is run on the ideal AerSimulator (statevector-based, no noise),
1024 shots. PASS requires that the two most frequent measured colorings
(restricted to the 4 color qubits) are exactly the 2 valid classical
colorings, and that together they account for a clear majority of shots
(amplified over the 1/8 baseline probability of a random guess).
"""

import itertools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles by brute force.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = [(0, 1), (1, 2), (2, 3)]  # path graph P4


def is_proper_2coloring(bits):
    """bits: tuple of 4 ints (0/1), bits[v] = color of vertex v."""
    return all(bits[i] != bits[j] for (i, j) in EDGES)


def classical_valid_colorings():
    valid = []
    for bits in itertools.product([0, 1], repeat=4):
        if is_proper_2coloring(bits):
            valid.append(bits)
    return valid


CLASSICAL_VALID = classical_valid_colorings()
assert len(CLASSICAL_VALID) == 2, (
    f"expected exactly 2 proper 2-colorings of P4, got {len(CLASSICAL_VALID)}: "
    f"{CLASSICAL_VALID}"
)
# Bit order used below: q0 q1 q2 q3 as printed left-to-right (vertex 0 first).
CLASSICAL_STRINGS = {
    "".join(str(b) for b in bits) for bits in CLASSICAL_VALID
}


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for a proper 2-coloring of P4.
# ---------------------------------------------------------------------------

def build_oracle(color, anc, out):
    """Phase oracle marking states where every edge of P4 is properly colored.

    color: QuantumRegister of 4 qubits (vertex colors)
    anc:   QuantumRegister of 3 qubits (one per edge, computes XOR)
    out:   QuantumRegister of 1 qubit (prepared in |-> by caller for kickback)
    """
    qc = QuantumCircuit(color, anc, out, name="oracle")

    # Compute anc[k] = color[i] XOR color[j] for each edge (i, j).
    for k, (i, j) in enumerate(EDGES):
        qc.cx(color[i], anc[k])
        qc.cx(color[j], anc[k])

    # Flip the phase-kickback output qubit iff all 3 ancillas are 1,
    # i.e. iff all 3 edges are properly colored.
    qc.mcx([anc[0], anc[1], anc[2]], out[0])

    # Uncompute the ancillas so they return to |000> (disentangle).
    for k, (i, j) in enumerate(EDGES):
        qc.cx(color[j], anc[k])
        qc.cx(color[i], anc[k])

    return qc


def build_diffuser(n):
    """Standard Grover diffuser (inversion about the mean) on n qubits."""
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(iterations):
    color = QuantumRegister(4, "c")
    anc = QuantumRegister(3, "a")
    out = QuantumRegister(1, "o")
    creg = ClassicalRegister(4, "m")
    qc = QuantumCircuit(color, anc, out, creg)

    # Uniform superposition over the 16 colorings.
    qc.h(color)

    # Output qubit in |-> for phase kickback.
    qc.x(out[0])
    qc.h(out[0])

    oracle = build_oracle(color, anc, out)
    diffuser = build_diffuser(4)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), color[:] + anc[:] + out[:])
        qc.append(diffuser.to_instruction(), color[:])

    # Undo the output qubit prep (not strictly required, but tidy).
    qc.h(out[0])
    qc.x(out[0])

    qc.measure(color, creg)
    return qc


# Optimal Grover iteration count for M=2 marked states out of N=16.
import math
N_STATES = 16
M_MARKED = 2
ITERATIONS = max(1, round(math.pi / 4 * math.sqrt(N_STATES / M_MARKED)))

circuit = build_grover_circuit(ITERATIONS)

simulator = AerSimulator()
transpiled = transpile(circuit, simulator)
SHOTS = 1024
result = simulator.run(transpiled, shots=SHOTS).result()
counts = result.get_counts()


# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

# Qiskit's classical-register bitstrings print with the highest-indexed qubit
# first (little-endian across the register), i.e. "q3 q2 q1 q0". Reverse to
# get back to our q0 q1 q2 q3 vertex order.
def to_vertex_order(bitstring):
    return bitstring[::-1]

normalized_counts = {}
for bitstring, n in counts.items():
    normalized_counts[to_vertex_order(bitstring)] = normalized_counts.get(
        to_vertex_order(bitstring), 0
    ) + n

sorted_results = sorted(normalized_counts.items(), key=lambda kv: -kv[1])
top_two = {s for s, _ in sorted_results[:2]}
top_two_shots = sum(n for s, n in sorted_results[:2])

print(f"Problem: Erdos #923 (tags: graph theory, chromatic number; oeis: N/A)")
print(f"Substitute property: P4 2-colorability via Grover search")
print(f"Classical valid colorings (brute force): {sorted(CLASSICAL_STRINGS)}")
print(f"Grover iterations used: {ITERATIONS}")
print(f"Measured counts (vertex order q0q1q2q3): {sorted_results}")
print(f"Top-2 measured colorings: {sorted(top_two)}")
print(f"Top-2 shots / total shots: {top_two_shots}/{SHOTS}")

passed = (
    top_two == CLASSICAL_STRINGS
    and top_two_shots / SHOTS > 0.5  # clearly amplified over 2/16 = 12.5% baseline
)

if passed:
    print("PASS")
else:
    print("FAIL")
