"""
Erdos problem #927 -- quantum-testable sequence attempt (honest best-effort, with a
documented limitation).

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 927"):
    prize: "no"
    informal_status: disproved (Lean) (last_update 2026-06-07)
    oeis: ["possible"]
    tags: ["graph theory"]

LIMITATION -- read before trusting the "OEIS id" framing:
The yaml's oeis field for problem 927 is the literal placeholder string
"possible", not a real OEIS sequence id (compare e.g. problem 928, which has a
genuine id like "A006530"). There is therefore no actual OEIS sequence to
derive a finite computable property from for this problem. Per the task's own
fallback instructions, this script does NOT fabricate a fake OEIS-grounded
property. Instead, using the one real piece of metadata available -- the tag
"graph theory" -- it builds a genuine, self-contained finite graph-theory
search problem (triangle-free labeled graphs on 4 vertices) and solves it with
a real Grover search circuit on the ideal AerSimulator. This is a generic
graph-theory demonstration in the spirit of the problem's tag, NOT a
verification of problem 927's actual mathematical content, and NOT tied to
any specific OEIS sequence. ran_ok/verified_against_classical below describe
only this substitute construction, honestly.

Classical property being tested:
    Search space: all 2^6 = 64 labeled graphs on the 4 vertices {0,1,2,3},
    represented as a 6-bit string over the edges
        (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
    in that fixed order (bit i = 1 means that edge is present).
    Marked set M = { graphs that are triangle-free, i.e. contain none of the
    4 possible triangles on {0,1,2,3} as a subgraph }. This is computed
    directly here in Python from first principles (brute-force enumeration of
    all 64 graphs, checking each of the 4 possible triangles), independent of
    any quantum computation.

Quantum circuit:
    A genuine Grover search over the 6-qubit, 64-state search space. The oracle
    is a diagonal phase-flip unitary built directly from the classically
    computed marked set M (a real, if precomputed, oracle -- not a lookup of
    the final answer), paired with the standard Grover diffuser, iterated the
    optimal number of times for |M|/64. The circuit is run on AerSimulator and
    the most frequently measured bitstring is checked against the classically
    computed marked set M.

PASS/FAIL: the script prints PASS if the most-probable measured outcome is
indeed a member of the classically computed triangle-free set M, else FAIL.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation of the marked set (first principles, no quantum).
# ---------------------------------------------------------------------------

VERTICES = (0, 1, 2, 3)
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # bit i <-> EDGES[i]
N_EDGES = len(EDGES)
N_STATES = 2 ** N_EDGES  # 64
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # the 4 possible triangles


def edge_index(u, v):
    e = (min(u, v), max(u, v))
    return EDGES.index(e)


TRIANGLE_EDGE_IDX = [
    tuple(edge_index(a, b) for a, b in itertools.combinations(tri, 2))
    for tri in TRIANGLES
]


def is_triangle_free(bits):
    """bits: length-6 tuple of 0/1, bits[i] = presence of EDGES[i].
    True iff no triangle among the 4 possible triangles is fully present."""
    for tri_idxs in TRIANGLE_EDGE_IDX:
        if all(bits[i] for i in tri_idxs):
            return False
    return True


def int_to_bits(n):
    return tuple((n >> i) & 1 for i in range(N_EDGES))


marked = []
for n in range(N_STATES):
    if is_triangle_free(int_to_bits(n)):
        marked.append(n)

M = len(marked)
assert M > 0 and M < N_STATES, "sanity check: nontrivial marked set"
print(f"Classical enumeration: {M} of {N_STATES} labeled graphs on K4 are triangle-free.")

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle as a diagonal unitary from the marked set.
# ---------------------------------------------------------------------------

diag = np.ones(N_STATES, dtype=complex)
for idx in marked:
    diag[idx] = -1.0
oracle_op = Operator(np.diag(diag))

n_qubits = N_EDGES  # 6

oracle_circ = QuantumCircuit(n_qubits, name="Oracle")
oracle_circ.append(oracle_op, range(n_qubits))

# ---------------------------------------------------------------------------
# 3. Standard Grover diffuser (inversion about the mean).
# ---------------------------------------------------------------------------


def diffuser(n):
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


diff_circ = diffuser(n_qubits)

# ---------------------------------------------------------------------------
# 4. Assemble full Grover circuit with the optimal iteration count.
# ---------------------------------------------------------------------------

iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M)))
print(f"Running Grover search with {iterations} iteration(s) over {n_qubits} qubits.")

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    qc.append(oracle_circ.to_instruction(), range(n_qubits))
    qc.append(diff_circ.to_instruction(), range(n_qubits))
qc.measure(range(n_qubits), range(n_qubits))

# ---------------------------------------------------------------------------
# 5. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
result = backend.run(tqc, shots=4096).result()
counts = result.get_counts()

# Qiskit bit ordering: classical bit 0 (qubit 0) is the rightmost character.
best_bitstring = max(counts, key=counts.get)
best_int = int(best_bitstring[::-1], 2)  # reverse to match qubit-index order

success_shots = sum(c for bs, c in counts.items() if int(bs[::-1], 2) in marked)
total_shots = sum(counts.values())
print(f"Most frequent measured state: {best_bitstring} -> graph index {best_int}, "
      f"count {counts[best_bitstring]}/{total_shots}")
print(f"Fraction of shots landing on a triangle-free graph: {success_shots / total_shots:.3f}")

quantum_found_marked = best_int in marked

if quantum_found_marked:
    print("PASS: Grover search's top outcome is a classically verified triangle-free graph.")
else:
    print("FAIL: Grover search's top outcome is NOT triangle-free.")
