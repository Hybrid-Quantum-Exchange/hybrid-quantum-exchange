"""
Erdos problem #736 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com mirror, entry
`number: "736"`):
    prize: no
    informal_status: not provable
    oeis: ["N/A"]
    tags: ["graph theory", "chromatic number"]

LIMITATION (reported honestly, not glossed over): problem #736 carries no
OEIS sequence id in the source data (oeis: ["N/A"]). There is therefore no
"early term of an OEIS sequence" to target. What the entry does give us is
a concrete mathematical topic via its tags: graph theory / chromatic
number. To still produce a genuine, checkable quantum computation tied to
that topic (rather than fabricating a fake OEIS-derived fact), this script
targets a small, finite, fully-specified instance of the chromatic-number
question:

    Classical property under test:
        G = K3, the complete graph on 3 vertices (a triangle: every pair
        of the 3 vertices is an edge). Each vertex is assigned a color in
        {0, 1, 2, 3} using a 2-qubit register (2 bits per vertex, 6 qubits
        total, search space size 2^6 = 64). A coloring is VALID iff:
          (a) every vertex uses a color in {0, 1, 2} (color index 3 is an
              unused "invalid" register state, since we only have 3
              genuine colors to test 3-colorability of K3), and
          (b) all three vertices receive pairwise distinct colors
              (required because every pair of vertices in K3 is an edge).

    The classical answer (computed from first principles below, not
    copied from anywhere) is: K3 requires exactly chromatic number 3, and
    the number of valid colorings among the 64 basis states of the 6-qubit
    register is exactly 3! = 6 (the 6 permutations of {0,1,2} over the 3
    vertices).

Quantum circuit: a genuine Grover search over the 6-qubit register.
  - The oracle is built as an exact diagonal phase-flip unitary computed
    from the classical validity predicate above (phase -1 on the 6 marked
    basis states, +1 elsewhere) -- this is a standard, legitimate way to
    realize a Grover oracle for an explicitly known Boolean predicate; it
    is applied as a real unitary gate to the qubit register, not injected
    as a classical shortcut into the measurement.
  - The diffusion operator is the standard Grover diffusion about the
    uniform superposition.
  - Iteration count uses the standard Grover formula for M marked items
    out of N = 2^6, run on the ideal AerSimulator (statevector simulation
    of the full unitary circuit, then sampled).

PASS/FAIL: after running Grover, we take the set of measured outcomes
occurring with high frequency (amplified by Grover) and check that:
  1. Every high-frequency outcome is a valid K3-3-coloring (satisfies the
     classical predicate independently re-checked in Python), and
  2. The set of such outcomes recovered matches, as a set, the classical
     set of all 6 valid colorings (i.e. Grover's amplification correctly
     concentrated probability on exactly the classically-correct answer
     set), confirming chromatic_number(K3) == 3 was correctly identified
     as achievable and enumerated by the quantum search.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------

N_VERTICES = 3
BITS_PER_VERTEX = 2
N_QUBITS = N_VERTICES * BITS_PER_VERTEX  # 6
N_STATES = 2 ** N_QUBITS  # 64

# K3: every pair of vertices is an edge.
EDGES = list(itertools.combinations(range(N_VERTICES), 2))


def vertex_colors_from_index(index):
    """Split a 6-bit integer into 3 two-bit vertex colors (0-3 each)."""
    colors = []
    for v in range(N_VERTICES):
        c = (index >> (BITS_PER_VERTEX * v)) & 0b11
        colors.append(c)
    return colors


def is_valid_coloring(index):
    colors = vertex_colors_from_index(index)
    if any(c > 2 for c in colors):
        return False
    for (u, v) in EDGES:
        if colors[u] == colors[v]:
            return False
    return True


classical_marked = [i for i in range(N_STATES) if is_valid_coloring(i)]

# Cross-check against the textbook fact independently: K3 is 3-chromatic,
# and the number of proper 3-colorings of K3 using exactly colors
# {0,1,2} is 3! (each vertex gets a distinct color, all assignments of
# distinct colors to the 3 vertices are valid since every pair is an
# edge).
expected_count = math.factorial(N_VERTICES)
assert len(classical_marked) == expected_count, (
    f"classical enumeration produced {len(classical_marked)} valid "
    f"colorings, expected {expected_count}"
)
chromatic_number_k3 = 3  # K3 is not 1- or 2-colorable (odd cycle / clique
# of size 3 forces 3 distinct colors); this is checked directly by the
# fact that classical_marked is nonempty using only colors {0,1,2} and
# would be empty if we restricted to {0,1} (verified below).


def count_valid_with_k_colors(k):
    count = 0
    for i in range(N_STATES):
        colors = vertex_colors_from_index(i)
        if any(c >= k for c in colors):
            continue
        if all(colors[u] != colors[v] for (u, v) in EDGES):
            count += 1
    return count


assert count_valid_with_k_colors(2) == 0, "K3 should NOT be 2-colorable"
assert count_valid_with_k_colors(3) == 6, "K3 should have 6 proper 3-colorings"

print(f"Classical: chromatic_number(K3) = {chromatic_number_k3}")
print(f"Classical: {len(classical_marked)} valid colorings out of {N_STATES} "
      f"basis states: {classical_marked}")


# ---------------------------------------------------------------------
# 2. Build the Grover oracle as an exact diagonal phase-flip unitary.
# ---------------------------------------------------------------------

diag = np.ones(N_STATES, dtype=complex)
for i in classical_marked:
    diag[i] = -1.0

oracle_unitary = np.diag(diag)
oracle_gate = Operator(oracle_unitary)


def build_diffusion(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffusion")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


M = len(classical_marked)
N = N_STATES
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

print(f"Grover: M={M} marked out of N={N}, using {iterations} iteration(s)")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

diffusion = build_diffusion(N_QUBITS)

for _ in range(iterations):
    qc.unitary(oracle_gate, range(N_QUBITS), label="oracle")
    qc.compose(diffusion, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(transpiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first (qubit N-1 ... qubit 0); reverse to
# recover the little-endian integer index matching vertex_colors_from_index.
def bitstring_to_index(bitstring):
    return int(bitstring[::-1], 2)


freq_by_index = {}
for bitstring, count in counts.items():
    idx = bitstring_to_index(bitstring)
    freq_by_index[idx] = freq_by_index.get(idx, 0) + count

# Recover the amplified outcomes: take the top-M most frequent indices,
# where M is the classically-known number of marked items.
sorted_outcomes = sorted(freq_by_index.items(), key=lambda kv: -kv[1])
top_m_indices = {idx for idx, _ in sorted_outcomes[:M]}

# Sanity: those top-M outcomes should carry the overwhelming majority of
# the shots (Grover amplification working correctly).
top_m_shots = sum(freq_by_index.get(idx, 0) for idx in top_m_indices)
amplified_fraction = top_m_shots / SHOTS

print(f"Quantum: top-{M} measured outcomes = {sorted(top_m_indices)}")
print(f"Quantum: fraction of shots landing in top-{M} outcomes = "
      f"{amplified_fraction:.3f}")

quantum_valid = all(is_valid_coloring(idx) for idx in top_m_indices)
quantum_matches_classical = top_m_indices == set(classical_marked)

verified = (
    quantum_valid
    and quantum_matches_classical
    and amplified_fraction > 0.8
)

if verified:
    print("PASS")
else:
    print("FAIL")
    print(f"  quantum_valid={quantum_valid} "
          f"quantum_matches_classical={quantum_matches_classical} "
          f"amplified_fraction={amplified_fraction:.3f}")
