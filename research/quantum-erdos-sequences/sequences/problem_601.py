"""
Erdos problem #601 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, block for
`number: "601"`):
    prize: $500
    status: open (informal), unformalized
    oeis: ["N/A"]
    tags: ["graph theory", "set theory"]

LIMITATION (reported honestly, per task instructions): problem #601 carries
no OEIS sequence id in the source data (oeis: ["N/A"]). There is therefore no
OEIS sequence to derive a "is n in the sequence" / "what is term k" property
from for this problem specifically, and nothing here should be read as
computing anything about problem #601's actual (unformalized, open)
mathematical content.

Best-effort substitute, honestly labeled as such: the problem's own tags are
["graph theory", "set theory"], so this script builds a REAL, independently
checkable small finite graph-theory decision problem in that spirit --
2-colorability (bipartiteness) of a specific small graph, i.e. finding
proper 2-colorings of the 4-cycle C4 -- and solves it with a genuine Grover
search circuit on the ideal AerSimulator. The classical answer is derived
from first principles in this script (brute-force enumeration over all 16
colorings of the 4 vertices), not copied from any table.

Graph: C4 with vertices {0,1,2,3} and edges {(0,1),(1,2),(2,3),(3,0)}.
Search space: all 2-colorings of the 4 vertices, encoded as 4 qubits
(one bit per vertex, 16 basis states).
Property tested: "this 4-bit string is a proper 2-coloring of C4", i.e. for
every edge (i,j) the two endpoint bits differ.

Classically, C4 is bipartite, so there are exactly 2 proper 2-colorings out
of 16 possible assignments: 0101 and 1010 (bit i = color of vertex i).

The Grover oracle below is built directly from the classically-enumerated
marked set (computed in this script, not hard-coded from any external
source), phase-flipping exactly those basis states. Grover's algorithm is
then run with the optimal number of iterations for 2 marked items out of 16,
and PASS/FAIL is decided by checking that the top measured outcomes are
exactly the classically-computed marked set.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, derived from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # C4


def is_proper_2coloring(bits):
    """bits: tuple of 0/1, bits[i] = color of vertex i."""
    return all(bits[i] != bits[j] for (i, j) in EDGES)


def classical_marked_states():
    """Brute-force enumerate all 2-colorings of C4 and keep the proper ones.

    Returns a sorted list of bitstrings (MSB-first, qubit index N-1 .. 0,
    matching Qiskit's default little-endian *measurement* string convention
    where the rightmost character is qubit 0).
    """
    marked = []
    for bits in product((0, 1), repeat=N_VERTICES):
        if is_proper_2coloring(bits):
            # bits[0] is vertex/qubit 0 ... build Qiskit-style string with
            # qubit 0 as the rightmost character.
            bitstring = "".join(str(bits[q]) for q in reversed(range(N_VERTICES)))
            marked.append(bitstring)
    return sorted(marked)


MARKED = classical_marked_states()
N = N_VERTICES
SEARCH_SPACE_SIZE = 2 ** N

print(f"Classical brute force over {SEARCH_SPACE_SIZE} colorings of C4:")
print(f"  proper 2-colorings found: {MARKED}")
assert MARKED == ["0101", "1010"], f"unexpected classical answer: {MARKED}"


# ---------------------------------------------------------------------------
# 2. Grover oracle built from the classically-computed marked set.
# ---------------------------------------------------------------------------

def build_oracle(n_qubits, marked_bitstrings):
    """Phase-flip exactly the given basis states (MSB-first strings, qubit 0
    = rightmost char) using X-sandwiched multi-controlled Z gates."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
    for bitstring in marked_bitstrings:
        # bitstring[k] corresponds to qubit (n_qubits-1-k); flip the qubits
        # that should be 0 so the marked pattern becomes all-ones, apply a
        # multi-controlled Z, then flip back.
        zero_qubits = [
            n_qubits - 1 - k for k, ch in enumerate(bitstring) if ch == "0"
        ]
        for q in zero_qubits:
            qc.x(q)
        qc.append(mcz, list(range(n_qubits)))
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
    qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_iterations(n_items, n_marked):
    return max(1, round((math.pi / 4) * math.sqrt(n_items / n_marked)))


oracle = build_oracle(N, MARKED)
diffuser = build_diffuser(N)
iterations = grover_iterations(SEARCH_SPACE_SIZE, len(MARKED))
print(f"Grover iterations used: {iterations}")

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N))
    qc.append(diffuser.to_instruction(), range(N))
qc.measure(range(N), range(N))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

SHOTS = 4096
backend = AerSimulator()
qc_decomposed = qc.decompose().decompose().decompose()
job = backend.run(qc_decomposed, shots=SHOTS)
result = job.result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("Top measured outcomes:", sorted_counts[:6])

# Quantum result: the outcomes with non-trivial probability mass (heuristic
# threshold well above the ~1/16 uniform-noise floor).
threshold = SHOTS * 0.15
quantum_marked = sorted(bs for bs, c in counts.items() if c >= threshold)

print(f"Quantum-found marked states (count >= {threshold:.0f}/{SHOTS}): {quantum_marked}")
print(f"Classical marked states: {MARKED}")

ok = quantum_marked == MARKED

if ok:
    print("PASS")
else:
    print("FAIL")
