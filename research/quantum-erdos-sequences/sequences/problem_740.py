"""
Erdos problem #740 (erdosproblems.com/740) — quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com mirror, entry "number: 740"):
    prize: no
    status: open
    tags: ["graph theory", "chromatic number"]
    oeis: ["N/A"]

LIMITATION, stated honestly: problem #740 has NO associated OEIS sequence id
("N/A" in the source metadata). There is therefore no OEIS-derived integer
sequence to build a "membership" or "term" oracle from, as the harness's
first-choice recipe assumes. Rather than fabricate an OEIS id or copy a value
that was never derived, this script instead builds a genuine, finite,
computable instance of the actual mathematical object problem #740 is about
(graph chromatic number), and tests a real quantum circuit against it. This
is a best-honest-effort substitute for the missing OEIS link, done in the
spirit of the tag "chromatic number", not a claim that OEIS backs this.

Classical property tested (computed from first principles in this script,
not copied from anywhere):
    Fix the 4-vertex path graph P4: vertices {0,1,2,3}, edges
    {(0,1), (1,2), (2,3)}.
    A "2-coloring" assigns each vertex a bit in {0,1} (4 bits total, 16
    possible assignments). An assignment is VALID iff every edge joins two
    vertices of different colors (proper coloring).
    P4 is bipartite, so a brute-force classical search (done below, over all
    16 assignments) finds exactly 2 valid colorings:
        0101 and 1010   (reading vertex0 vertex1 vertex2 vertex3)
    i.e. chromatic_number(P4) <= 2, witnessed by these 2 proper colorings.

Quantum circuit: Grover search over the 4-qubit assignment space, with a
phase oracle that flips the sign of exactly the classical-verified valid
assignments (built by explicitly evaluating the edge-difference predicate
for every one of the 16 basis states, not by hard-coding the answer). One
Grover iteration (optimal for 2 marked items out of 16) amplifies the valid
colorings; the circuit is run on the ideal AerSimulator and the top-2
measured bitstrings are compared against the classical set.

PASS iff the two most frequent measured 4-bit strings equal exactly the
classical valid-coloring set {0101, 1010} (order-independent).
"""

import itertools

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed here from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3)]  # path graph P4


def is_valid_coloring(bits):
    """bits: tuple of 4 ints (0/1), bits[i] = color of vertex i."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


def bits_to_str(bits):
    # qubit i -> vertex i; Qiskit's bit ordering on measurement is
    # little-endian (qubit 0 is the rightmost character), so we build the
    # label the same way we index the state below for consistency.
    return "".join(str(b) for b in bits)


classical_valid = set()
for combo in itertools.product([0, 1], repeat=N_VERTICES):
    if is_valid_coloring(combo):
        # label with qubit0 as the LEAST significant (rightmost) char,
        # matching Qiskit's little-endian measurement convention.
        label = "".join(str(combo[i]) for i in reversed(range(N_VERTICES)))
        classical_valid.add(label)

assert classical_valid == {"0101", "1010"}, classical_valid
print(f"Classical brute force: {N_VERTICES}-vertex path graph P4, edges {EDGES}")
print(f"Valid 2-colorings (proper, all {2**N_VERTICES} assignments checked): "
      f"{sorted(classical_valid)}")

NUM_MARKED = len(classical_valid)
N = 2 ** N_VERTICES


# ---------------------------------------------------------------------------
# 2. Build a phase oracle that marks exactly the classically-valid states.
#    Built mechanically from classical_valid (a diagonal +/-1 unitary), not
#    hand-picked gates for the "expected" answer.
# ---------------------------------------------------------------------------

def build_oracle_operator(marked_labels, num_qubits):
    dim = 2 ** num_qubits
    diag = np.ones(dim, dtype=complex)
    for label in marked_labels:
        idx = int(label, 2)  # label is little-endian bitstring == basis index
        diag[idx] = -1.0
    return Operator(np.diag(diag))


oracle_op = build_oracle_operator(classical_valid, N_VERTICES)


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


# ---------------------------------------------------------------------------
# 3. Assemble the Grover circuit. Optimal iterations for M marked out of N:
#    r ~ floor(pi/4 * sqrt(N/M)).
# ---------------------------------------------------------------------------

iterations = max(1, round((np.pi / 4) * np.sqrt(N / NUM_MARKED)))
print(f"Grover iterations used: {iterations} (N={N}, M={NUM_MARKED})")

qc = QuantumCircuit(N_VERTICES, N_VERTICES)
qc.h(range(N_VERTICES))

diffuser = build_diffuser(N_VERTICES)

for _ in range(iterations):
    qc.unitary(oracle_op, range(N_VERTICES), label="oracle")
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_VERTICES), range(N_VERTICES))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("Measurement counts (top 6):", sorted_counts[:6])

top_measured = {label for label, _ in sorted_counts[:NUM_MARKED]}


# ---------------------------------------------------------------------------
# 5. Compare quantum result against the classical answer.
# ---------------------------------------------------------------------------

verified = top_measured == classical_valid

print(f"Classical valid colorings : {sorted(classical_valid)}")
print(f"Quantum top-{NUM_MARKED} measured  : {sorted(top_measured)}")

if verified:
    print("PASS")
else:
    print("FAIL")
