#!/usr/bin/env python3
"""
Erdos problem #129 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, manman4/erdosproblems
data/problems.yaml, entry "number: '129'"):
    prize: no
    status: open
    tags: ["graph theory", "ramsey theory"]
    oeis: ["possible"]
    comments: "ambiguous statement"

IMPORTANT LIMITATION, stated honestly up front: problem #129's yaml entry
does NOT carry a real OEIS sequence id. The oeis field literally contains
the string "possible" -- not an A-number -- and the entry's own comment
says the problem's statement is "ambiguous". There is therefore no OEIS
sequence to derive a property from for this problem, and no way to honor
the "identify a property from its OEIS id(s)" instruction literally: there
is no id.

Best-effort substitute actually used here: the problem is tagged
"graph theory" / "ramsey theory". In lieu of a fabricated OEIS value, this
script tests a genuine, small, finite, classically-checkable fact from
Ramsey theory that is representative of the problem's tags and is
completely honest about not being tied to problem #129's specific
(ambiguous, prize:no) statement:

    Classical property under test
    ------------------------------
    R(3,3) = 6, i.e. K_4 (4 vertices, 6 edges) CAN be 2-edge-colored with
    no monochromatic triangle, while K_6 cannot. We take the search space
    of all 2^6 = 64 edge-2-colorings of K_4 (one bit per edge) and define
    the marked set M = colorings with no monochromatic triangle among the
    four triangles of K_4. The classical answer, computed here from first
    principles by brute-force enumeration (no OEIS lookup, no literature
    value), is |M| and the explicit list of marked colorings.

    The quantum circuit is a real Grover search: a 6-qubit oracle marking
    exactly the elements of M (built as an exact diagonal phase oracle
    from the brute-force truth table, not hand-tuned), the standard
    Grover diffuser, and the classically-optimal number of iterations.
    Run on the ideal AerSimulator, the circuit must concentrate
    measurement probability onto the marked set. PASS/FAIL compares the
    quantum-simulator's most-probable outcomes against the independently
    computed classical marked set.

This is a genuine Grover search over a real, verifiable, finite
combinatorial fact (Ramsey-flavored, matching the problem's tags); it is
NOT a claim about problem #129's own (ambiguous) statement, and this
script does not pretend otherwise.
"""

import math
import sys
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate, DiagonalGate
from qiskit_aer import AerSimulator

ERDOS_PROBLEM_NUMBER = 129
OEIS_IDS_USED: list[str] = []  # none: the yaml entry's oeis field is "possible", not a real A-number
CLASSICAL_PROPERTY = (
    "2-edge-colorings of K_4 (6 edges, one bit per edge) containing no "
    "monochromatic triangle"
)

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(combinations(VERTICES, 2))          # 6 edges, index 0..5
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

TRIANGLES = list(combinations(VERTICES, 3))       # 4 triangles
TRIANGLE_EDGE_IDX = []
for tri in TRIANGLES:
    a, b, c = tri
    idxs = (
        EDGE_INDEX[tuple(sorted((a, b)))],
        EDGE_INDEX[tuple(sorted((a, c)))],
        EDGE_INDEX[tuple(sorted((b, c)))],
    )
    TRIANGLE_EDGE_IDX.append(idxs)


def has_monochromatic_triangle(coloring_bits: int) -> bool:
    """coloring_bits: 6-bit int, bit i = color (0/1) of EDGES[i]."""
    for (i, j, k) in TRIANGLE_EDGE_IDX:
        ci = (coloring_bits >> i) & 1
        cj = (coloring_bits >> j) & 1
        ck = (coloring_bits >> k) & 1
        if ci == cj == ck:
            return True
    return False


N_QUBITS = 6
N_STATES = 2 ** N_QUBITS  # 64

MARKED = [x for x in range(N_STATES) if not has_monochromatic_triangle(x)]
CLASSICAL_COUNT = len(MARKED)

print(f"Erdos problem #{ERDOS_PROBLEM_NUMBER} quantum-testable instance")
print(f"OEIS ids used: {OEIS_IDS_USED!r} (none -- see docstring: yaml oeis field is 'possible', not an id)")
print(f"Classical property: {CLASSICAL_PROPERTY}")
print(f"Classical brute-force count of marked colorings |M| = {CLASSICAL_COUNT} / {N_STATES}")
print(f"Marked colorings (first few): {MARKED[:8]}{'...' if len(MARKED) > 8 else ''}")

assert 0 < CLASSICAL_COUNT < N_STATES, "degenerate search space; Grover needs a nontrivial marked set"

# ---------------------------------------------------------------------------
# 2. Quantum circuit: exact Grover search with a diagonal phase oracle
#    built directly from the brute-force truth table above.
# ---------------------------------------------------------------------------


def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Exact diagonal phase oracle: flips sign on exactly the `marked` basis states."""
    diag = np.ones(2 ** n_qubits, dtype=complex)
    for m in marked:
        diag[m] = -1.0
    qc = QuantumCircuit(n_qubits, name="oracle")
    qc.append(DiagonalGate(list(diag)), list(range(n_qubits)))
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(N_QUBITS, MARKED)
diffuser = build_diffuser(N_QUBITS)

# Classically-optimal number of Grover iterations for this M, N.
theta = math.asin(math.sqrt(CLASSICAL_COUNT / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit order: rightmost measured-classical-bit is qubit 0 -> matches our
# EDGES[i] <-> qubit i convention when we read the bitstring reversed.
def bitstring_to_int(bs: str) -> int:
    return int(bs[::-1], 2)

marked_set = set(MARKED)
total_marked_shots = sum(c for bs, c in counts.items() if bitstring_to_int(bs) in marked_set)
marked_fraction = total_marked_shots / SHOTS

top_outcomes = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
print("Top measured outcomes (bitstring, count, int, is_marked):")
for bs, c in top_outcomes:
    x = bitstring_to_int(bs)
    print(f"  {bs}  count={c:5d}  x={x:2d}  marked={x in marked_set}")

print(f"Fraction of shots landing on a marked (classically verified) state: {marked_fraction:.3f}")

# ---------------------------------------------------------------------------
# 4. PASS/FAIL: the quantum search must concentrate probability on the
#    classically-computed marked set well above the uniform baseline
#    (uniform would give CLASSICAL_COUNT / N_STATES), and the single most
#    probable measured outcome must itself be a classically-verified
#    marked state.
# ---------------------------------------------------------------------------

uniform_baseline = CLASSICAL_COUNT / N_STATES
most_probable_bitstring, _ = top_outcomes[0]
most_probable_is_marked = bitstring_to_int(most_probable_bitstring) in marked_set

ran_ok = True
verified_against_classical = (
    most_probable_is_marked and marked_fraction > 3 * uniform_baseline
)

if verified_against_classical:
    print("PASS: Grover search concentrated on classically-verified marked states "
          f"({marked_fraction:.3f} vs uniform baseline {uniform_baseline:.3f}).")
    sys.exit(0)
else:
    print("FAIL: Grover search did not concentrate on the classically-verified marked set.")
    sys.exit(1)
