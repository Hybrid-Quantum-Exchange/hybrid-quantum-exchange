"""
Erdos problem #24 -- quantum-testable instance.

Source metadata (erdosproblems.com data, `data/problems.yaml` in the
manman4/erdosproblems clone, entry `number: "24"`):
    tags: ["graph theory"]
    oeis: ["possible"]
    status: proved (Lean)

LIMITATION, stated honestly up front: the `oeis` field for problem #24 in
the source data is the literal string "possible", not a real OEIS sequence
id (e.g. "A000041"). There is no genuine OEIS sequence attached to this
problem in the data, and no problem statement/description field is present
in the YAML entry either -- only the tag "graph theory" and the proof
status. So this script cannot test "membership in the OEIS sequence for
problem 24," because no such sequence id exists to test against.

Best-honest-effort substitute, staying inside the declared tag ("graph
theory"): a small, finite, genuinely computable graph-theory decision
property closely tied to the kind of extremal/coloring statements Erdos
problems in this tag area are about --

    PROPERTY TESTED: proper 2-colorings of the 4-cycle graph C4
    (vertices 0,1,2,3; edges (0,1),(1,2),(2,3),(3,0)).
    A 2-coloring is an assignment of one bit (color) to each of the 4
    vertices. It is "proper" iff every edge joins two differently-colored
    vertices (no monochromatic edge). Since C4 is bipartite, the classical
    count of proper 2-colorings out of all 2^4 = 16 assignments is known
    (and is computed from scratch below, not copied from anywhere): the
    only proper 2-colorings are the two alternating colorings
    0101 and 1010, so there are exactly 2 solutions among 16 candidates.

APPROACH: Grover's search. 4 qubits encode the 4-bit color assignment.
A reversible oracle (built from CNOTs into 4 "edge-check" ancillas, an
AND-into-flag ancilla, then phase kickback via the flag ancilla in the
|-> state, then full uncomputation) marks the 2 proper colorings with a
phase flip. One diffusion (inversion-about-mean) round follows -- for a
16-item search space with 2 marked items, a single Grover iteration is
close to optimal (optimal iteration count round(pi/4 * sqrt(16/2)) = 2,
so we run 2 iterations for a strong amplitude boost). The circuit is run
on the ideal AerSimulator and the most-frequent measured bitstrings are
compared against the classical brute-force answer.

PASS/FAIL: the script prints PASS iff the set of bitstrings receiving the
top measurement counts (a number of them equal to the true solution
count) exactly equals the classical solution set.
"""

import itertools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # the 4-cycle C4


def is_proper_2coloring(bits):
    """bits: tuple of 4 ints (0/1), one color per vertex."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


classical_solutions = sorted(
    "".join(str(b) for b in bits)
    for bits in itertools.product([0, 1], repeat=N_VERTICES)
    if is_proper_2coloring(bits)
)

print(f"Classical brute force over all {2**N_VERTICES} assignments of C4:")
print(f"  proper 2-colorings found: {classical_solutions}")
assert classical_solutions == ["0101", "1010"], "unexpected classical result"
NUM_SOLUTIONS = len(classical_solutions)

# ---------------------------------------------------------------------------
# 2. Grover oracle for "is this a proper 2-coloring of C4".
#
# Qubit layout (bit order v0 v1 v2 v3, Qiskit reports qubit 0 as the
# rightmost character of the classical-register bitstring):
#   v[0..3]  : the 4 color qubits (the search register)
#   e[0..3]  : one ancilla per edge, set to 1 iff that edge's endpoints
#              differ (computed with a pair of CNOTs into a fresh ancilla)
#   flag     : set to 1 iff ALL edge ancillas are 1 (multi-controlled X)
# Phase kickback: flag ancilla is prepared in |-> before the oracle and
# left there after, so whenever flag would be flipped to 1 the marked
# states instead pick up a -1 phase. Everything but v[] is then uncomputed.
# ---------------------------------------------------------------------------

v = QuantumRegister(N_VERTICES, "v")
e = QuantumRegister(len(EDGES), "e")
flag = QuantumRegister(1, "flag")
c = ClassicalRegister(N_VERTICES, "c")


def build_oracle(qc):
    # compute edge-difference ancillas: e[i] = v[u] XOR v[uu]
    for i, (u, w) in enumerate(EDGES):
        qc.cx(v[u], e[i])
        qc.cx(v[w], e[i])
    # flip flag (currently |-> ) iff all edge ancillas are 1 -> phase kick
    qc.mcx(list(e), flag[0])
    # uncompute edge ancillas
    for i, (u, w) in enumerate(EDGES):
        qc.cx(v[w], e[i])
        qc.cx(v[u], e[i])


def build_diffuser(qc, reg):
    qc.h(reg)
    qc.x(reg)
    qc.h(reg[-1])
    qc.mcx(list(reg[:-1]), reg[-1])
    qc.h(reg[-1])
    qc.x(reg)
    qc.h(reg)


qc = QuantumCircuit(v, e, flag, c)

# init: uniform superposition over the 4 color qubits, flag ancilla in |->
qc.h(v)
qc.x(flag)
qc.h(flag)

ITERATIONS = 2  # round(pi/4 * sqrt(16/2)) = 2, near-optimal for 2 marked/16
for _ in range(ITERATIONS):
    build_oracle(qc)
    build_diffuser(qc, v)

qc.measure(v, c)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
SHOTS = 4096
result = backend.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("\nTop measured bitstrings (Qiskit little-endian, qubit v0 is rightmost char):")
for bitstring, n in sorted_counts[:6]:
    print(f"  {bitstring}: {n}")

top_bitstrings = sorted(bs for bs, _ in sorted_counts[:NUM_SOLUTIONS])

# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer and report.
# ---------------------------------------------------------------------------

print(f"\nClassical solutions : {classical_solutions}")
print(f"Quantum top-{NUM_SOLUTIONS} states : {top_bitstrings}")

if top_bitstrings == classical_solutions:
    print("PASS")
else:
    print("FAIL")
