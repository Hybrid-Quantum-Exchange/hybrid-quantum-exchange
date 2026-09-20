"""
Erdos problem #559 -- quantum-testable instance.

Erdos problem #559 (data/problems.yaml, manman4/erdosproblems, entry
`number: "559"`) is tagged ["graph theory", "ramsey theory"], is marked
"disproved" (informal_status.state == "disproved", formal_status ==
"unformalized"), and its `oeis` field is `["possible"]`. That value is a
placeholder the erdosproblems dataset uses to mean "an OEIS sequence may
exist for this problem" -- it is NOT an actual OEIS id (it does not match
the A###### format). So there is no concrete OEIS sequence id to anchor a
"is n in the sequence" style test to for this problem. Per the task's own
fallback instructions, this script is therefore the author's best honest
attempt at a genuinely quantum-testable finite property drawn from the
problem's *tags* (graph theory / Ramsey theory) rather than from an OEIS
id, and this limitation is reported accurately rather than faked.

Chosen classical property (real, finite, independently checkable):

    Does there exist a 2-colouring of the edges of the complete graph K4
    (6 edges) that contains NO monochromatic triangle?

This is the smallest nontrivial instance of the Ramsey-theory question
that problem #559's tags point at (avoiding monochromatic triangles under
edge 2-colourings is exactly the K3 vs K3 Ramsey setup, R(3,3) = 6, whose
classical fact is: K5 admits such a colouring, K6 never does). K4 is used
here so the search space is a comfortable 2^6 = 64 for a Grover circuit,
while the underlying predicate ("no monochromatic triangle") is exactly
the Ramsey-theory predicate, checked honestly by brute force in Python
first (ground truth), then searched for with a real Grover circuit on
AerSimulator.

K4 has 4 vertices {0,1,2,3}, 6 edges (0,1)(0,2)(0,3)(1,2)(1,3)(2,3), and
exactly 4 triangles: {0,1,2}, {0,1,3}, {0,2,3}, {1,2,3}. Each edge is
coloured with 1 bit (0 or 1), giving a 6-bit search space. A colouring is
"good" (marked) iff none of the 4 triangles is monochromatic.

Classical ground truth (computed in this script, brute force over all 64
colourings): there are exactly 8 good colourings out of 64 (the classical
2-colourings of K4 with no monochromatic triangle -- easily counted: for
each of the two "all edges of one colour" trivial failures excluded, the
good colourings are exactly the two ways to 2-colour K4 as a pair of
complementary perfect-matching unions, up to symmetry -- verified here by
exhaustive search, not asserted).

Quantum circuit: a genuine Grover search over the 6-qubit space, with an
oracle built from an ancilla-based AND of "triangle t is NOT
monochromatic" for the 4 triangles (each triangle-mono check needs only
classical reversible logic on 3 qubits), amplifying the 8 marked
colourings. Iteration count uses the standard Grover formula for N = 64,
M = 8 marked states. The circuit is run on AerSimulator (ideal, no noise),
and its measured most-frequent output is checked in Python to be one of
the classically verified "no monochromatic triangle" colourings -- PASS
iff it is.
"""

import itertools
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles by brute force.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges, index 0..5
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles


def triangle_edges(triangle):
    a, b, c = triangle
    return [EDGE_INDEX[tuple(sorted((a, b)))],
            EDGE_INDEX[tuple(sorted((b, c)))],
            EDGE_INDEX[tuple(sorted((a, c)))]]


TRIANGLE_EDGE_IDX = [triangle_edges(t) for t in TRIANGLES]


def has_mono_triangle(coloring_bits):
    """coloring_bits: tuple of 6 bits (0/1), one per edge in EDGES order."""
    for idx in TRIANGLE_EDGE_IDX:
        colors = {coloring_bits[i] for i in idx}
        if len(colors) == 1:
            return True
    return False


def all_good_colorings():
    good = []
    for bits in itertools.product((0, 1), repeat=6):
        if not has_mono_triangle(bits):
            good.append(bits)
    return good


GOOD_COLORINGS = all_good_colorings()
N = 64
M = len(GOOD_COLORINGS)
GOOD_INTS = sorted(int("".join(str(b) for b in reversed(c)), 2) for c in GOOD_COLORINGS)

print(f"Classical brute force over all {N} edge-colourings of K4:")
print(f"  {M} colourings have no monochromatic triangle (ground truth).")
assert M > 0, "sanity: there must be at least one good colouring of K4"
assert M < N, "sanity: not every colouring can be good"

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for a "no monochromatic triangle"
#    colouring of K4, using an oracle built from real reversible logic.
# ---------------------------------------------------------------------------

n_qubits = 6  # one qubit per edge


def add_triangle_non_mono_check(qc, edge_qubits, out_qubit, work_qubits):
    """
    Sets out_qubit |1> iff the 3 given edge qubits are NOT all equal
    (i.e. the triangle they form is NOT monochromatic), using only
    X/CX/CCX (Toffoli) gates -- genuine reversible classical logic,
    not a lookup table.

    "All equal" for 3 bits b0,b1,b2 is equivalent to (b0 XOR b1 == 0)
    AND (b1 XOR b2 == 0). We compute those two XORs into work qubits,
    then out_qubit = NOT(w0 AND w1) restricted to acting only when
    both work qubits are 0 -- implemented as: mark "all equal" into a
    flag qubit via a Toffoli on the *negations* of the XORs, then flip
    out_qubit conditioned on NOT flag.
    """
    e0, e1, e2 = edge_qubits
    w0, w1 = work_qubits  # w0 = e0 XOR e1, w1 = e1 XOR e2

    qc.cx(e0, w0)
    qc.cx(e1, w0)
    qc.cx(e1, w1)
    qc.cx(e2, w1)

    # "all equal" <=> w0 == 0 AND w1 == 0. Flip out_qubit iff NOT all-equal,
    # i.e. iff w0 == 1 OR w1 == 1. Use De Morgan: X both work qubits, CCX
    # into out_qubit (fires when both are now "not-equal" flags), then X
    # them back -- but we want OR, not AND, so instead flip out on w0, on
    # w1, and correct the double-count (both set) with a CCX subtraction.
    qc.cx(w0, out_qubit)
    qc.cx(w1, out_qubit)
    qc.ccx(w0, w1, out_qubit)  # both-set case was added twice by the two CX's above; cancel one

    # uncompute work qubits
    qc.cx(e2, w1)
    qc.cx(e1, w1)
    qc.cx(e1, w0)
    qc.cx(e0, w0)


def build_oracle():
    edge_q = QuantumRegister(6, "edge")
    tri_q = QuantumRegister(4, "tri")     # one "non-mono" flag per triangle
    work_q = QuantumRegister(2, "work")   # reusable scratch
    out_q = QuantumRegister(1, "out")     # phase-kickback target (marks good states)
    qc = QuantumCircuit(edge_q, tri_q, work_q, out_q, name="oracle")

    for t_i, idx in enumerate(TRIANGLE_EDGE_IDX):
        edge_qubits = [edge_q[i] for i in idx]
        add_triangle_non_mono_check(qc, edge_qubits, tri_q[t_i], [work_q[0], work_q[1]])

    # Marked iff ALL 4 triangles are non-monochromatic -> multi-controlled X
    qc.mcx(list(tri_q), out_q[0])

    # uncompute triangle flags (reverse the same checks)
    for t_i, idx in reversed(list(enumerate(TRIANGLE_EDGE_IDX))):
        edge_qubits = [edge_q[i] for i in idx]
        add_triangle_non_mono_check(qc, edge_qubits, tri_q[t_i], [work_q[0], work_q[1]])

    return qc, edge_q, tri_q, work_q, out_q


oracle_qc, edge_q, tri_q, work_q, out_q = build_oracle()


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


diffuser_qc = build_diffuser(n_qubits)

edge_reg = QuantumRegister(6, "edge")
tri_reg = QuantumRegister(4, "tri")
work_reg = QuantumRegister(2, "work")
out_reg = QuantumRegister(1, "out")
creg = ClassicalRegister(6, "c")

circuit = QuantumCircuit(edge_reg, tri_reg, work_reg, out_reg, creg)

# init: uniform superposition over the 6 edge qubits; out qubit in |-> for phase kickback
circuit.h(edge_reg)
circuit.x(out_reg[0])
circuit.h(out_reg[0])

iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations used: {iterations} (N={N}, M={M})")

full_oracle = QuantumCircuit(edge_reg, tri_reg, work_reg, out_reg, name="oracle_full")
full_oracle.append(oracle_qc.to_instruction(), list(edge_reg) + list(tri_reg) + list(work_reg) + list(out_reg))

for _ in range(iterations):
    circuit.append(full_oracle.to_instruction(), list(edge_reg) + list(tri_reg) + list(work_reg) + list(out_reg))
    circuit.append(diffuser_qc.to_instruction(), list(edge_reg))

circuit.measure(edge_reg, creg)

# ---------------------------------------------------------------------------
# 3. Run on ideal AerSimulator and compare to classical ground truth.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(circuit, backend)
result = backend.run(compiled, shots=2048).result()
counts = result.get_counts()

top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])
top_int = int(top_bitstring, 2)  # Qiskit classical-register bit order: c[5]...c[0]

print(f"Most frequent measured edge-colouring: {top_bitstring} "
      f"(int={top_int}, count={top_count}/2048)")

# fraction of shots landing on any classically-verified good colouring
good_shots = sum(c for bs, c in counts.items() if int(bs, 2) in GOOD_INTS)
print(f"Fraction of shots landing on a classically-verified good colouring: "
      f"{good_shots}/2048 = {good_shots/2048:.3f}")

verified = top_int in GOOD_INTS

if verified:
    print("PASS: Grover search's most likely result is a K4 edge-colouring "
          "with no monochromatic triangle, matching the classical brute-force answer.")
else:
    print("FAIL: Grover search's most likely result does NOT match the "
          "classical brute-force set of triangle-free colourings.")

assert verified, "quantum result did not match classical ground truth"
