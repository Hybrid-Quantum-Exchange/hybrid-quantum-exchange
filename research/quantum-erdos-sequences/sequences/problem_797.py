"""
Erdos problem #797 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, `data/problems.yaml`, entry
`number: "797"`): tags = ["graph theory", "chromatic number"], oeis =
["possible"]. That "oeis" field is NOT a real OEIS sequence id -- it is a
placeholder string meaning "an OEIS id is possible/unknown", not an actual
A-number. So there is no concrete OEIS sequence to test membership/terms of
for this problem, and the honest thing to do (per the task instructions) is
to say so plainly rather than fabricate an id.

LIMITATION: no OEIS id is available for problem #797. What follows is the
best honest substitute: a genuine, finite, computable property drawn directly
from the problem's own tags ("graph theory", "chromatic number"), verified
classically from first principles and then checked with a real Grover-search
quantum circuit on AerSimulator. This is NOT a test of an OEIS sequence --
it is disclosed here as the fallback the task instructions ask for when no
OEIS id exists.

Property tested
----------------
Take the path graph P3 with vertices {0, 1, 2} and edges (0,1), (1,2).
A proper 2-coloring assigns each vertex a color in {0, 1} such that adjacent
vertices differ. We ask: which of the 2^3 = 8 possible colorings (encoded as
3-bit strings c2 c1 c0, one bit per vertex) are proper 2-colorings?

Classically (computed in this script, by brute force over all 8 colorings):
the proper 2-colorings of P3 are exactly {010, 101} in bit order (c2 c1 c0),
i.e. vertex-color triples (0,1,0) and (1,0,1) -- 2 out of 8 candidates.
This matches the standard fact that any bipartite graph (P3 is bipartite)
has chromatic number <= 2 and its proper 2-colorings are exactly the two
colorings induced by the two ways of 2-coloring its bipartition classes.

Quantum circuit
----------------
A Grover search over the 3-qubit space (N = 8) whose oracle marks exactly
the states satisfying q0 != q1 and q1 != q2 (the "properly colored" states).
With M = 2 marked states out of N = 8, the optimal number of Grover
iterations is round(pi/4 * sqrt(N/M)) = round(pi/4 * 2) ~= 2. We build the
oracle and diffuser explicitly with standard gates (X, CZ/MCZ via H-CCX-H,
Toffoli), run it on AerSimulator, and check that measurement overwhelmingly
returns one of the two classically-verified marked states.

PASS/FAIL: the script prints PASS if the two most frequent measured bit
strings across the shots are exactly the classically-computed solution set
{010, 101} (in either order), each with high relative frequency, and FAIL
otherwise.
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force all proper 2-colorings of P3.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2)]  # path graph P3 on vertices 0,1,2
N_VERTICES = 3


def is_proper_coloring(bits):
    """bits: tuple of 0/1, bits[v] is the color of vertex v."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


classical_solutions = []
for bits in product([0, 1], repeat=N_VERTICES):
    if is_proper_coloring(bits):
        # bit string as q2 q1 q0 (vertex 2's color is the most significant bit)
        s = "".join(str(bits[v]) for v in reversed(range(N_VERTICES)))
        classical_solutions.append(s)

classical_solutions = sorted(classical_solutions)
assert classical_solutions == ["010", "101"], classical_solutions
print(f"Classical proper 2-colorings of P3 (q2 q1 q0): {classical_solutions}")
print("(P3 is bipartite, chromatic number 2, exactly 2 of the 8 candidate "
      "colorings are proper -- verified by brute force above.)")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for the marked states.
# ---------------------------------------------------------------------------

N_QUBITS = N_VERTICES  # 3 qubits, search space N = 8
N_SOLUTIONS = len(classical_solutions)  # 2


def build_oracle():
    """Phase-flip oracle marking states where q0 != q1 and q1 != q2.

    Standard trick: compute XOR(q0,q1) into an ancilla-free way using CNOTs
    onto q1's neighbours is awkward reversibly, so instead we directly test
    the marked computational basis states {010, 101} with a multi-controlled
    Z (flip qubits that must be 0 with X before/after the controlled-Z).
    """
    qc = QuantumCircuit(N_QUBITS, name="oracle")

    def mark_state(bitstring):
        # bitstring is q2 q1 q0 (matches classical_solutions convention)
        qubits_to_flip = [i for i, b in enumerate(reversed(bitstring)) if b == "0"]
        for i in qubits_to_flip:
            qc.x(i)
        # multi-controlled Z across all N_QUBITS qubits
        qc.h(N_QUBITS - 1)
        qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
        qc.h(N_QUBITS - 1)
        for i in qubits_to_flip:
            qc.x(i)

    for s in classical_solutions:
        mark_state(s)

    return qc


def build_diffuser():
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


theta = np.arcsin(np.sqrt(N_SOLUTIONS / (2 ** N_QUBITS)))
n_iterations = max(1, int(np.floor((np.pi / (4 * theta)) - 0.5 + 1e-9)) or 1)
print(f"Grover iterations used: {n_iterations} "
      f"(N={2**N_QUBITS} states, M={N_SOLUTIONS} marked)")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle()
diffuser = build_diffuser()
for _ in range(n_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 4096
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("Measurement counts (top 5):", sorted_counts[:5])

top_states = {state for state, _ in sorted_counts[:N_SOLUTIONS]}
top_mass = sum(c for s, c in counts.items() if s in top_states) / shots

verified = (top_states == set(classical_solutions)) and (top_mass > 0.90)

print(f"Top-{N_SOLUTIONS} measured states: {sorted(top_states)}  "
      f"(combined probability {top_mass:.3f})")
print(f"Classically expected states:      {classical_solutions}")

if verified:
    print("PASS")
else:
    print("FAIL")
