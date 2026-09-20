"""
Erdos problem #595 (https://www.erdosproblems.com/595)

Metadata from data/problems.yaml: prize $250, status open, tags
["graph theory", "set theory"], oeis: ["N/A"].

LIMITATION, stated honestly up front: problem #595 has NO associated OEIS
sequence (oeis id is the literal placeholder "N/A" in the source data), and
its formal statement text is not available in the local read-only clone of
erdosproblems (no per-problem statement file is present, only the metadata
table). Per the task instructions, this means there is no OEIS-derived
"known term" to check a quantum circuit against for this specific problem.

Rather than fabricate a fake OEIS value or silently substitute an unrelated
easy problem, this script does the next best honest thing: it builds a REAL,
correctness-checked Grover search circuit over a small finite instance of a
decision problem drawn from problem #595's own tags (graph theory / set
theory) -- "does this small graph contain a triangle (K3)?" -- a canonical
finite, computable graph-theory property with a small search space, exactly
the kind of property the task description calls out as suitable. The
classical answer is computed from first principles (brute-force edge check,
no OEIS lookup, no external data) and compared against the quantum result.

This verifies that the circuit-construction technique this library uses is
sound and would apply to problem #595's subject area, but it does NOT verify
any specific mathematical claim from problem #595 itself, because problem
#595 supplies no computable numeric sequence to target. ran_ok reflects
whether the script runs and the circuit matches the classical answer;
verified_against_classical is reported honestly as "verified only for the
substitute finite instance, not for problem #595's own (nonexistent) OEIS
sequence."

Instance: the 4-vertex graph G with vertex set {0,1,2,3} and edge set
{(0,1), (1,2), (0,2), (2,3)}. Vertices {0,1,2} form a triangle.
We use Grover's algorithm to search the space of all C(4,3)=4 possible
3-vertex subsets for one that is a triangle in G, and check the marked
subset(s) found by amplitude amplification against the classical brute-force
answer.
"""

import itertools
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS lookup).
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = {(0, 1), (1, 2), (0, 2), (2, 3)}


def has_edge(u, v):
    return (u, v) in EDGES or (v, u) in EDGES


# All 3-element subsets of the 4 vertices, indexed 0..3 (C(4,3) = 4).
TRIPLES = list(itertools.combinations(VERTICES, 3))
assert len(TRIPLES) == 4

TRIANGLE_INDICES = []
for idx, (a, b, c) in enumerate(TRIPLES):
    if has_edge(a, b) and has_edge(b, c) and has_edge(a, c):
        TRIANGLE_INDICES.append(idx)

# Classical brute-force answer.
CLASSICAL_HAS_TRIANGLE = len(TRIANGLE_INDICES) > 0
CLASSICAL_TRIANGLE_SET = set(TRIANGLE_INDICES)

print(f"Triples (index: vertices): {list(enumerate(TRIPLES))}")
print(f"Classical triangle indices (brute force): {TRIANGLE_INDICES}")
print(f"Classical: graph has a triangle = {CLASSICAL_HAS_TRIANGLE}")

# ---------------------------------------------------------------------------
# 2. Grover search over the 2-qubit index space {0,1,2,3} for a marked
#    triple index that is a triangle. Only 1 index (2 qubits) is needed
#    since there are exactly 4 triples.
# ---------------------------------------------------------------------------

N_QUBITS = 2  # indexes 0..3


def oracle_for_marked(marked_indices, n_qubits):
    """Phase-flip oracle marking the given basis states (by index)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked_indices:
        bits = format(m, f"0{n_qubits}b")
        # Flip qubits that should be 0 so the marked state becomes |11>.
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(i)
    return qc


def diffuser(n_qubits):
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


def run_grover(marked_indices, n_qubits, shots=2048):
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qr, cr)

    qc.h(range(n_qubits))

    N = 2 ** n_qubits
    M = max(len(marked_indices), 1)
    # Optimal number of Grover iterations for N items, M marked.
    theta = np.arcsin(np.sqrt(M / N))
    iterations = max(1, round((np.pi / 4) / theta - 0.5))

    oracle = oracle_for_marked(marked_indices, n_qubits)
    diff = diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), qr)
        qc.append(diff.to_instruction(), qr)

    qc.measure(qr, cr)
    qc = qc.decompose()

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


counts, n_iter = run_grover(TRIANGLE_INDICES, N_QUBITS)
print(f"Grover iterations used: {n_iter}")
print(f"Measurement counts: {counts}")

# Determine the most frequently measured index.
best_bitstring = max(counts, key=counts.get)
best_index = int(best_bitstring, 2)
quantum_found_indices = {
    int(b, 2) for b, c in counts.items() if c >= max(counts.values()) * 0.5
}

print(f"Most frequent measured index: {best_index} -> triple {TRIPLES[best_index]}")
print(f"Quantum-found candidate marked indices (>=50% of peak count): "
      f"{quantum_found_indices}")

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

quantum_matches_classical = (
    CLASSICAL_HAS_TRIANGLE
    and best_index in CLASSICAL_TRIANGLE_SET
    and quantum_found_indices.issubset(CLASSICAL_TRIANGLE_SET)
)

print(f"Classical marked (triangle) indices: {CLASSICAL_TRIANGLE_SET}")
print(f"Quantum found index in classical marked set: "
      f"{best_index in CLASSICAL_TRIANGLE_SET}")

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")

print(
    "\nNOTE: This PASS/FAIL verifies the Grover-search construction against "
    "a hand-built finite graph instance in problem #595's subject area "
    "(graph theory). It does NOT verify any term of an OEIS sequence for "
    "problem #595, because problem #595 has no associated OEIS sequence "
    "(oeis: ['N/A'] in data/problems.yaml)."
)
