"""
Erdos problem #618 — quantum-testable sequence attempt.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: '618'"):
    prize: no
    status: proved (Lean)
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION (read before trusting the PASS below):
Problem #618 has NO associated OEIS sequence id in the source data (oeis is
literally "N/A"). Per task instructions, when no OEIS id exists there is no
"sequence" to build a quantum-testable property from in the intended sense
(membership/term-of-sequence). This script is therefore a best-effort
fallback, not a genuine entry in a library of "quantum-testable OEIS
sequences": it uses the problem's only real piece of content — its tag,
"graph theory" — to build a small, finite, honestly-computable graph property
and verifies a real Grover search circuit against it. This is a legitimate
quantum computation, but it is NOT tied to any OEIS sequence for problem 618,
because none exists.

Chosen classical property (finite, computable, no OEIS involved):
    Fix the 5-vertex "house" graph (a 4-cycle 0-1-2-3-0 plus vertex 4
    attached to 0 and 1, i.e. a square with a triangular roof), a standard
    small test graph. Question: does this graph contain a triangle
    (a set of 3 mutually adjacent vertices)?
    We answer this classically by brute-force search over all C(5,3)=10
    vertex triples (computed in this script from first principles, no
    external data), and we answer it with a Grover search over the same
    10-element search space using a real Qiskit circuit on AerSimulator,
    then compare.

Grover search space encoding:
    The 10 unordered triples of {0,1,2,3,4} are indexed 0..9 and each index
    is represented by 4 qubits (2^4 = 16 >= 10). The oracle marks exactly
    the indices whose triple is a triangle in the graph. Grover's algorithm
    amplifies those marked basis states; measurement should return one of
    them with high probability. We compare the set of measured states
    (after enough shots) against the classically-computed triangle triples.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed here, from first principles).
# ---------------------------------------------------------------------------

# "House" graph on 5 vertices: square 0-1-2-3-0 plus roof vertex 4 attached
# to 0 and 1, forming a triangle 0-1-4.
EDGES = {(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)}


def is_edge(u, v):
    return (u, v) in EDGES or (v, u) in EDGES


VERTICES = [0, 1, 2, 3, 4]
ALL_TRIPLES = list(itertools.combinations(VERTICES, 3))  # 10 triples, index 0..9
assert len(ALL_TRIPLES) == 10

TRIANGLE_INDICES = []
for idx, (a, b, c) in enumerate(ALL_TRIPLES):
    if is_edge(a, b) and is_edge(b, c) and is_edge(a, c):
        TRIANGLE_INDICES.append(idx)

# Classical brute-force answer.
print("All vertex triples (index: triple):")
for idx, t in enumerate(ALL_TRIPLES):
    marker = "  <- triangle" if idx in TRIANGLE_INDICES else ""
    print(f"  {idx}: {t}{marker}")
print(f"Classical triangle indices: {TRIANGLE_INDICES}")

assert len(TRIANGLE_INDICES) > 0, "expected at least one triangle in the house graph"

# ---------------------------------------------------------------------------
# 2. Grover search over the 10 indices (4 qubits, states 0..15; 10..15 unused
#    and never marked, so they simply never contribute to the marked set).
# ---------------------------------------------------------------------------

N_QUBITS = 4
N_STATES = 2 ** N_QUBITS  # 16


def index_to_bits(idx, n=N_QUBITS):
    return [(idx >> k) & 1 for k in range(n)]  # little-endian: bit0 = qubit0


def build_oracle(marked_indices, n=N_QUBITS):
    """Phase oracle: flips sign of |idx> for idx in marked_indices."""
    qc = QuantumCircuit(n, name="oracle")
    for idx in marked_indices:
        bits = index_to_bits(idx, n)
        # Flip qubits that should be 0 so the marked pattern becomes all-1s.
        for q, b in enumerate(bits):
            if b == 0:
                qc.x(q)
        # Multi-controlled Z on all n qubits (phase flip on |11..1>).
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for q, b in enumerate(bits):
            if b == 0:
                qc.x(q)
    return qc


def build_diffuser(n=N_QUBITS):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover_circuit(marked_indices, n=N_QUBITS, n_states=N_STATES):
    m = len(marked_indices)
    # Optimal number of Grover iterations for amplitude amplification.
    theta = math.asin(math.sqrt(m / n_states))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5)) if theta > 0 else 1

    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    oracle = build_oracle(marked_indices, n)
    diffuser = build_diffuser(n)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n))
        qc.append(diffuser.to_gate(), range(n))

    qc.measure(range(n), range(n))
    return qc, iterations


qc, iters = grover_circuit(TRIANGLE_INDICES)
print(f"\nGrover circuit built with {iters} iteration(s) for "
      f"{len(TRIANGLE_INDICES)} marked state(s) out of {N_STATES}.")

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical register bit order in the returned bitstring is
# c[n-1] c[n-2] ... c[0] (most significant bit first, matching qubit n-1 ..
# qubit 0). Since we mapped qubit k -> bit k of idx (little-endian), we
# reverse the bitstring to recover idx directly.
def bitstring_to_index(bs):
    # Qiskit's returned bitstring is c[n-1]...c[0] left-to-right, and since
    # measure(range(n), range(n)) maps clbit i <- qubit i (weight 2^i), the
    # bitstring read as an ordinary binary number already equals idx.
    return int(bs, 2)

measured_indices = {}
for bitstring, count in counts.items():
    idx = bitstring_to_index(bitstring)
    measured_indices[idx] = measured_indices.get(idx, 0) + count

# Take the states that got meaningfully amplified (well above the uniform
# background of shots/16), sorted by count.
background = shots / N_STATES
amplified = sorted(
    (idx for idx, c in measured_indices.items() if c > 3 * background),
    key=lambda idx: -measured_indices[idx],
)

print(f"\nMeasured index counts (top by count): "
      f"{sorted(measured_indices.items(), key=lambda kv: -kv[1])[:6]}")
print(f"Amplified (quantum-found) triangle indices: {sorted(amplified)}")

# ---------------------------------------------------------------------------
# 3. Compare quantum result to classical ground truth.
# ---------------------------------------------------------------------------

quantum_set = set(amplified)
classical_set = set(TRIANGLE_INDICES)
verified = quantum_set == classical_set

print(f"\nClassical triangle index set: {sorted(classical_set)}")
print(f"Quantum (Grover-amplified) index set: {sorted(quantum_set)}")

if verified:
    print("PASS")
else:
    print("FAIL")
