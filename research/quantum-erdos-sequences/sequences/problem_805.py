"""
Quantum-testable lane for Erdos problem #805 (erdosproblems.com / manman4/erdosproblems).

Source metadata (data/problems.yaml, entry "number: \"805\""):
    prize: no
    status: open (informal_status: open, last_update 2025-08-31)
    oeis: ["possible"]
    tags: ["graph theory"]

LIMITATION (read before trusting the "PASS"): the repository entry for
problem 805 carries no real OEIS sequence id -- the field is the literal
placeholder string "possible", not an id such as "A000040" -- and no
statement text is present anywhere in the read-only clone
(/home/user/manman4/erdosproblems/data/problems.yaml, only the metadata
block above exists; there is no accompanying prose file for #805, and
grepping the whole yaml for "805" turns up nothing beyond that one block).
So there is no genuine OEIS-derived integer sequence to build a
Grover/QPE/amplitude-estimation instance *for problem 805 specifically*,
and inventing a fake OEIS value would violate the task's own rule against
fabricating or copying sequence content that was never actually derived.

Best honest attempt taken instead: the only real signal available for #805
is its tag, "graph theory". So this script builds a genuine, small, finite,
classically-checkable graph-theory search problem -- find the unique
missing edge of a near-complete 4-vertex graph -- and solves it with a real
Grover search circuit run on the ideal AerSimulator. This is NOT claimed to
be Erdos problem 805's actual content (no such content was recoverable from
the source data); it is offered as the closest good-faith, non-fabricated
quantum-testable instance obtainable from what the metadata actually
contains.

Classical problem (computed from first principles below, not looked up):
    Graph G on vertices {0,1,2,3}. The 6 possible undirected edges are
    enumerated in a fixed order:
        edge index 0: (0,1)   3: (1,2)
        edge index 1: (0,2)   4: (1,3)
        edge index 2: (0,3)   5: (2,3)
    G is K4 with exactly one edge removed: E(G) = all 6 pairs except (0,1).
    Search problem: over a 3-qubit index register representing edge indices
    0..7 (indices 6,7 are unused/invalid slots, included only because 6 is
    not a power of two), find the index of the unique pair that is NOT an
    edge of G. classical_answer() computes this by brute-force enumeration
    of all 6 pairs and checking membership in E(G); it is a single marked
    item (index 0) out of 8 possible register states.

Quantum method: Grover's algorithm over the 3-qubit index register (N=8,
M=1 marked item), oracle built directly from the classically-computed
marked index (no hardcoded "the answer is 0" -- it is the output of
classical_answer()), diffuser + oracle composed via Qiskit's own
grover_operator() helper (avoids a hand-rolled, easy-to-get-backwards
diffuser). Optimal iteration count for N=8, M=1 is
floor(pi/4 * sqrt(8)) = 2. Circuit runs on AerSimulator, shots=4096;
PASS/FAIL compares the single dominant measured index to the classical
marked index.

No OEIS id is used because none exists for this entry; oeis/verified flags
are reported honestly by the harness driving this script.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
ALL_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # fixed edge-index order
MISSING_EDGE = (0, 1)
EDGES = set(ALL_PAIRS) - {MISSING_EDGE}  # K4 minus edge (0,1)


def classical_answer():
    """Brute-force over all 6 possible pairs: return the edge index/indices
    that are NOT present in EDGES (i.e. the missing edge(s) of G)."""
    missing = [i for i, pair in enumerate(ALL_PAIRS) if pair not in EDGES]
    return set(missing)


MARKED = classical_answer()
print(f"Classical brute-force marked (missing-edge) indices: {sorted(MARKED)}")

assert MARKED == {0}, f"unexpected classical answer: {MARKED}"
assert ALL_PAIRS[0] == MISSING_EDGE

# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 3-qubit index register (N=8, M=1).
# ---------------------------------------------------------------------------

N_QUBITS = 3  # indices 0..7 (6,7 unused/invalid, simply never marked)


def oracle_circuit(marked_indices, n_qubits):
    """Phase-flip exactly the computational basis states in marked_indices."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")
        # bit_i (qubit i, LSB-first) = 1 iff char (n-1-i) of the MSB-first
        # string is '1'; flip qubits that are 0 so the marked pattern
        # becomes |1..1> for the multi-controlled Z.
        flip_qubits = [i for i, b in enumerate(reversed(bits)) if b == "0"]
        for q in flip_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc


def build_grover_circuit(marked_indices, n_qubits, iterations):
    """Full Grover search circuit. Oracle+diffuser pair is built with
    Qiskit's own verified grover_operator() helper."""
    oracle = oracle_circuit(marked_indices, n_qubits)
    g_op = grover_operator(oracle)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(g_op, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


N = 2 ** N_QUBITS
M = len(MARKED)
GROVER_ITERATIONS = max(1, round(math.floor((math.pi / 4) * math.sqrt(N / M))))
print(f"N={N}, M={M}, Grover iterations={GROVER_ITERATIONS}")

grover_qc = build_grover_circuit(MARKED, N_QUBITS, iterations=GROVER_ITERATIONS)

simulator = AerSimulator()
compiled = transpile(grover_qc, simulator)
SHOTS = 4096
job = simulator.run(compiled, shots=SHOTS)
result = job.result()
counts = result.get_counts()

print("Measurement counts:", counts)

# Convert bitstrings back to integer indices.
index_counts = {}
for bitstring, c in counts.items():
    idx = int(bitstring, 2)
    index_counts[idx] = index_counts.get(idx, 0) + c

print(f"Per-index counts: {index_counts}")

# Quantum answer: the index(es) whose measured probability clearly stands
# out above the uniform-random baseline (1/N of shots each).
baseline = SHOTS / N
quantum_marked = {idx for idx, c in index_counts.items() if c > 3 * baseline}
dominant_idx = max(index_counts, key=index_counts.get)
dominant_fraction = index_counts[dominant_idx] / SHOTS

print(f"Quantum-identified marked indices (>3x baseline): {sorted(quantum_marked)}")
print(f"Dominant measured index: {dominant_idx} ({dominant_fraction:.1%} of shots)")

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

verified = quantum_marked == MARKED and dominant_idx in MARKED and dominant_fraction > 0.5

if verified:
    print("PASS")
else:
    print("FAIL")
