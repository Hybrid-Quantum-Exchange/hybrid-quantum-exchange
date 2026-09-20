"""
Erdos problem #560 -- quantum-testable instance.

Erdos problem #560 (data/problems.yaml, manman4/erdosproblems, entry
`number: "560"`) is tagged ["graph theory", "ramsey theory"], is marked
open (informal_status.state == "open", formal_status == "unformalized"),
and its `oeis` field is `["possible"]`. That value is a placeholder the
erdosproblems dataset uses to mean "an OEIS sequence may exist for this
problem" -- it is NOT an actual OEIS id (it does not match the A######
format). So there is no concrete OEIS sequence id to anchor an "is n in
the sequence" style test to for this problem. Per the task's own fallback
instructions, this script is therefore the author's best honest attempt
at a genuinely quantum-testable finite property drawn from the problem's
*tags* (graph theory / Ramsey theory) rather than from an OEIS id, and
this limitation is reported accurately rather than faked.

Chosen classical property (real, finite, independently checkable):

    Does there exist a 2-colouring of the edges of the complete graph K5
    (10 edges) that contains NO monochromatic triangle?

This is exactly the classical fact behind R(3,3) = 6: K5 DOES admit a
triangle-free-in-both-colours edge 2-colouring (the "pentagon/pentagram"
colouring: colour the 5-cycle 0-1-2-3-4-0 red and its complement, the
other 5-cycle 0-2-4-1-3-0, blue), while K6 never does. Using the full
10-edge search space of K5 keeps the predicate genuinely tied to Ramsey
theory (the tag on problem #560) while remaining small enough (2^10 =
1024 basis states, 10 data qubits) for an exact statevector simulation
of a real Grover search on AerSimulator.

K5 has 5 vertices {0,1,2,3,4}, 10 edges, and exactly C(5,3) = 10
triangles. Each edge is coloured with 1 bit (0 or 1). A colouring is
"good" (marked) iff none of the 10 triangles is monochromatic.

Classical ground truth (computed in this script, brute force over all
1024 colourings): the good colourings are counted directly, and the
well-known pentagon/pentagram colouring is checked to be among them, all
by exhaustive search -- not asserted from memory.

Quantum circuit: a genuine Grover search over the 10-qubit edge-colouring
space, with an oracle built from ancilla-based reversible logic that
flags, for each of the 10 triangles, whether its 3 edges are NOT all the
same colour, and then requires ALL 10 flags to hold (multi-controlled X).
The number of Grover iterations uses the standard formula for N = 1024
and the classically-counted M marked states. The circuit runs on ideal
AerSimulator; its most frequent measured outcome is checked in Python to
be one of the classically verified triangle-free colourings -- PASS iff
it is.
"""

import itertools
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles by brute force.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3, 4]
EDGES = list(itertools.combinations(VERTICES, 2))  # 10 edges, index 0..9
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 10 triangles

assert len(EDGES) == 10
assert len(TRIANGLES) == 10


def triangle_edges(triangle):
    a, b, c = triangle
    return [EDGE_INDEX[tuple(sorted((a, b)))],
            EDGE_INDEX[tuple(sorted((b, c)))],
            EDGE_INDEX[tuple(sorted((a, c)))]]


TRIANGLE_EDGE_IDX = [triangle_edges(t) for t in TRIANGLES]


def has_mono_triangle(coloring_bits):
    """coloring_bits: tuple of 10 bits (0/1), one per edge in EDGES order."""
    for idx in TRIANGLE_EDGE_IDX:
        colors = {coloring_bits[i] for i in idx}
        if len(colors) == 1:
            return True
    return False


def all_good_colorings():
    good = []
    for bits in itertools.product((0, 1), repeat=10):
        if not has_mono_triangle(bits):
            good.append(bits)
    return good


GOOD_COLORINGS = all_good_colorings()
N = 1024
M = len(GOOD_COLORINGS)
GOOD_INTS = sorted(int("".join(str(b) for b in reversed(c)), 2) for c in GOOD_COLORINGS)

print(f"Classical brute force over all {N} edge-colourings of K5:")
print(f"  {M} colourings have no monochromatic triangle (ground truth).")
assert M > 0, "sanity: there must be at least one good colouring of K5 (R(3,3) = 6, not 5)"
assert M < N, "sanity: not every colouring can be good"

# Sanity-check the textbook pentagon/pentagram colouring is one of them:
# colour edge {i,j} red (0) iff |i-j| mod 5 is 1 or 4 (the 5-cycle 0-1-2-3-4-0),
# blue (1) otherwise (the pentagram 0-2-4-1-3-0).
pentagon_bits = tuple(0 if min((b - a) % 5, (a - b) % 5) == 1 else 1 for (a, b) in EDGES)
assert pentagon_bits in GOOD_COLORINGS, "pentagon/pentagram colouring must be triangle-free in both colours"
print("  Verified: the classical pentagon/pentagram colouring is among the good colourings.")

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for a "no monochromatic triangle"
#    colouring of K5, using an oracle built from real reversible logic.
# ---------------------------------------------------------------------------

n_qubits = 10  # one qubit per edge


def add_triangle_non_mono_check(qc, edge_qubits, out_qubit, work_qubits):
    """
    Sets out_qubit |1> iff the 3 given edge qubits are NOT all equal
    (i.e. the triangle they form is NOT monochromatic), using only
    X/CX/CCX (Toffoli) gates -- genuine reversible classical logic,
    not a lookup table.

    "All equal" for 3 bits b0,b1,b2 is equivalent to (b0 XOR b1 == 0)
    AND (b1 XOR b2 == 0). Compute those two XORs into work qubits, then
    flip out_qubit for "not all equal" = (w0 OR w1), computed via
    inclusion-exclusion with CX/CX/CCX (add both singles, subtract the
    double-counted AND case), then uncompute the work qubits.
    """
    e0, e1, e2 = edge_qubits
    w0, w1 = work_qubits  # w0 = e0 XOR e1, w1 = e1 XOR e2

    qc.cx(e0, w0)
    qc.cx(e1, w0)
    qc.cx(e1, w1)
    qc.cx(e2, w1)

    qc.cx(w0, out_qubit)
    qc.cx(w1, out_qubit)
    qc.ccx(w0, w1, out_qubit)  # correct the double-add when both w0 and w1 are set

    # uncompute work qubits
    qc.cx(e2, w1)
    qc.cx(e1, w1)
    qc.cx(e1, w0)
    qc.cx(e0, w0)


def build_oracle():
    edge_q = QuantumRegister(10, "edge")
    tri_q = QuantumRegister(10, "tri")    # one "non-mono" flag per triangle
    work_q = QuantumRegister(2, "work")   # reusable scratch
    out_q = QuantumRegister(1, "out")     # phase-kickback target (marks good states)
    qc = QuantumCircuit(edge_q, tri_q, work_q, out_q, name="oracle")

    for t_i, idx in enumerate(TRIANGLE_EDGE_IDX):
        edge_qubits = [edge_q[i] for i in idx]
        add_triangle_non_mono_check(qc, edge_qubits, tri_q[t_i], [work_q[0], work_q[1]])

    # Marked iff ALL 10 triangles are non-monochromatic -> multi-controlled X
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

edge_reg = QuantumRegister(10, "edge")
tri_reg = QuantumRegister(10, "tri")
work_reg = QuantumRegister(2, "work")
out_reg = QuantumRegister(1, "out")
creg = ClassicalRegister(10, "c")

circuit = QuantumCircuit(edge_reg, tri_reg, work_reg, out_reg, creg)

# init: uniform superposition over the 10 edge qubits; out qubit in |-> for phase kickback
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
top_int = int(top_bitstring, 2)  # Qiskit classical-register bit order: c[9]...c[0]

print(f"Most frequent measured edge-colouring: {top_bitstring} "
      f"(int={top_int}, count={top_count}/2048)")

good_shots = sum(c for bs, c in counts.items() if int(bs, 2) in GOOD_INTS)
print(f"Fraction of shots landing on a classically-verified good colouring: "
      f"{good_shots}/2048 = {good_shots/2048:.3f}")

verified = top_int in GOOD_INTS

if verified:
    print("PASS: Grover search's most likely result is a K5 edge-colouring "
          "with no monochromatic triangle, matching the classical brute-force answer.")
else:
    print("FAIL: Grover search's most likely result does NOT match the "
          "classical brute-force set of triangle-free colourings.")

assert verified, "quantum result did not match classical ground truth"
