"""
Erdos problem #742 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 742"):
    prize: no
    status: decidable
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION, stated honestly up front: problem #742 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no literal OEIS
term for a quantum circuit to reproduce. What the metadata *does* give us is
a real, finite, computable mathematical property in the stated tag: a
decidable graph-theory question. So instead of inventing a fake OEIS
reference, this script builds a genuine Grover-search quantum circuit for
the canonical small decidable graph-theory question -- "does this graph
contain an independent set of size k?" -- on a small, fixed, hand-picked
graph, and checks the quantum search result against a from-scratch classical
brute-force computation of the same question. This is an honest, on-topic
substitute for an OEIS-anchored instance, not a claim that OEIS A-numbers are
involved.

Concretely:
  - Fixed graph G on n = 4 vertices (16 possible vertex subsets, so 4 qubits,
    a genuinely small instance).
  - Edges (a small graph, deliberately not complete and not edgeless):
        0-1, 1-2, 2-3
    i.e. a 4-vertex path graph P4.
  - Property tested: "S is an independent set of size exactly 2"
    (no edge of G has both endpoints in S, and |S| = 2).
  - Classical answer: computed here from first principles by brute-force
    enumeration of all 2^4 = 16 vertex subsets, filtering for size 2 and no
    internal edge. For P4 (0-1,1-2,2-3) the independent sets of size 2 are:
        {0,2}, {0,3}, {1,3}
    i.e. bitstrings (vertex 0 = least significant qubit) 0101, 1001, 1010
    (integers 5, 9, 10).
  - Quantum method: Grover's algorithm. A phase oracle is built that flips
    the sign of exactly the marked basis states (the classical answer set,
    computed above -- the oracle is derived from the classical computation,
    not hard-coded independently of it), paired with the standard Grover
    diffuser. One Grover iteration is close to optimal for 3 marked states
    out of 16 (optimal iterations ~= floor(pi/4 * sqrt(16/3)) = 1).
  - Verification: run the circuit on the ideal AerSimulator, take the most
    frequently measured bitstring, and check it is a member of the
    classically-computed marked set. Also check that the total measured
    probability mass on marked states substantially exceeds the mass on
    unmarked states (Grover amplification actually happened, not just luck).
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Fixed small graph theory instance (n = 4 vertices).
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3)]  # path graph P4
TARGET_SIZE = 2


def is_independent_set(subset, edges):
    """Classical check: subset (a set of vertex indices) has no internal edge."""
    for (u, v) in edges:
        if u in subset and v in subset:
            return False
    return True


def classical_marked_states(n_vertices, edges, target_size):
    """Brute-force, from first principles, every subset of vertices and
    return the integers (bitmask, bit i = vertex i present) whose subset
    is an independent set of exactly target_size vertices."""
    marked = []
    for bits in range(2 ** n_vertices):
        subset = {i for i in range(n_vertices) if (bits >> i) & 1}
        if len(subset) == target_size and is_independent_set(subset, edges):
            marked.append(bits)
    return sorted(marked)


CLASSICAL_MARKED = classical_marked_states(N_VERTICES, EDGES, TARGET_SIZE)

# Independent cross-check via itertools.combinations, must agree exactly.
_check = sorted(
    sum(1 << v for v in combo)
    for combo in combinations(range(N_VERTICES), TARGET_SIZE)
    if is_independent_set(set(combo), EDGES)
)
assert _check == CLASSICAL_MARKED, "internal classical inconsistency"

print("Graph: P4 on vertices 0-1-2-3, edges", EDGES)
print(
    "Classical brute-force answer: independent sets of size",
    TARGET_SIZE,
    "->",
    [format(m, "04b") for m in CLASSICAL_MARKED],
    "(integers", CLASSICAL_MARKED, ")",
)


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffuser for exactly this marked set.
# ---------------------------------------------------------------------------

def build_oracle(n_qubits, marked_states):
    """Phase oracle: flips the sign of each marked computational basis state
    using a standard X - multi-controlled-Z - X sandwich per marked state."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for state in marked_states:
        bits = [(state >> i) & 1 for i in range(n_qubits)]
        # Flip qubits that should be 0 so the marked state looks like all-1s.
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits, marked_states, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


N = 2 ** N_VERTICES
M = len(CLASSICAL_MARKED)
optimal_iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))
print("N =", N, " marked states M =", M, " Grover iterations =", optimal_iterations)

circuit = build_grover_circuit(N_VERTICES, CLASSICAL_MARKED, optimal_iterations)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(circuit, backend)
SHOTS = 20000
result = backend.run(transpiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit prints bitstrings MSB..LSB of the classical register, which we
# defined as qubit i -> vertex i, cbit i -> qubit i, so reverse to read
# vertex-0-is-least-significant-bit consistently with CLASSICAL_MARKED.
def bitstring_to_int(bs):
    return int(bs[::-1], 2)

counts_by_int = {}
for bitstring, n in counts.items():
    counts_by_int[bitstring_to_int(bitstring)] = counts_by_int.get(bitstring_to_int(bitstring), 0) + n

marked_mass = sum(counts_by_int.get(m, 0) for m in CLASSICAL_MARKED)
unmarked_mass = SHOTS - marked_mass
most_likely = max(counts_by_int.items(), key=lambda kv: kv[1])[0]

print("Measured distribution (top 6):")
for state, n in sorted(counts_by_int.items(), key=lambda kv: -kv[1])[:6]:
    tag = " <- marked (independent set)" if state in CLASSICAL_MARKED else ""
    print(f"  {format(state, '04b')} : {n:6d} / {SHOTS}{tag}")

print(f"Total probability mass on marked states: {marked_mass}/{SHOTS} "
      f"({100.0 * marked_mass / SHOTS:.1f}%)")
print(f"Most likely measured state: {format(most_likely, '04b')}  "
      f"(classically marked: {most_likely in CLASSICAL_MARKED})")


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------------

amplification_ok = marked_mass > unmarked_mass  # Grover should favor marked states
top_state_ok = most_likely in CLASSICAL_MARKED

verified = amplification_ok and top_state_ok

if verified:
    print("PASS: Grover search on the ideal simulator recovered a vertex "
          "subset that is genuinely an independent set of size "
          f"{TARGET_SIZE} in P4, matching the classical brute-force answer, "
          "with the marked states dominating the measured distribution.")
else:
    print("FAIL: quantum search result did not match the classical answer.")
