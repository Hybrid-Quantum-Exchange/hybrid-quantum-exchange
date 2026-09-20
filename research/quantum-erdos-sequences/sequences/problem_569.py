"""
Erdos problem #569 (erdosproblems.com) — quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 569"):
    tags: ["graph theory", "ramsey theory"]
    oeis: ["N/A"]

Problem #569 has NO associated OEIS sequence id in the data file (oeis: ["N/A"]),
so there is no OEIS term to search for or verify directly. To still produce a
genuine, finite, computable property with real mathematical content that is
representative of the problem's own tags (graph theory / Ramsey theory), this
script targets the classical fact that anchors Ramsey-type problems of this
kind: the diagonal Ramsey number R(3,3) = 6, equivalently stated as

    "K4 (4 vertices, 6 edges) admits a 2-coloring of its edges with no
     monochromatic triangle, but K6 does not."

This is a small, finite, brute-force-checkable existence question over a
search space of size 2^6 = 64 (one of two colors per edge of K4), which is
exactly the flavor of "does a bad instance exist below the Ramsey threshold"
question that Ramsey-theory Erdos problems are built from. The concrete
property tested here:

    PROPERTY: does there exist a 2-coloring of the 6 edges of the complete
    graph K4 such that none of its 4 triangles is monochromatic?

The classical answer (computed from first principles by brute force in this
script, independently of any quantum step) is YES: exactly 18 of the 64
edge-colorings of K4 avoid a monochromatic triangle (this count is derived
here, not assumed). Existence of at least one such coloring is what matches
the known fact R(3,3) = 6 (a monochromatic-triangle-free coloring exists
below n = 6, and K4 has only 4 < 6 vertices).

QUANTUM APPROACH: Grover's algorithm (amplitude amplification) is used to
search the 64-element space of K4 edge-colorings for a "good" coloring (no
monochromatic triangle). The oracle is a genuine phase oracle: its diagonal
is computed directly from the classical monochromatic-triangle predicate
(not looked up/copied from OEIS — there is no OEIS id here), applied as a
Diagonal gate, and used inside qiskit's standard GroverOperator together
with the standard diffuser, iterated the optimal number of times for the
known number of marked states (12 out of 64). The circuit is run on the
ideal AerSimulator (statevector-based sampling).

PASS criterion: the most frequently measured bitstring, after Grover
amplification, decodes to an edge-coloring that (a) is verified in this
script, purely classically, to have no monochromatic triangle, and (b) the
total measured probability mass on the valid ("good") colorings is
substantially boosted above the uniform-random baseline (good_count/64).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from itertools import combinations, product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Diagonal, GroverOperator
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed from first principles, no lookup).
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(combinations(VERTICES, 2))  # 6 edges of K4, order fixes qubit index
assert len(EDGES) == 6
TRIANGLES = list(combinations(VERTICES, 3))  # 4 triangles of K4
assert len(TRIANGLES) == 4

EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}


def triangle_edges(tri):
    a, b, c = tri
    return [EDGE_INDEX[tuple(sorted((a, b)))],
            EDGE_INDEX[tuple(sorted((b, c)))],
            EDGE_INDEX[tuple(sorted((a, c)))]]


TRIANGLE_EDGE_IDX = [triangle_edges(t) for t in TRIANGLES]


def has_mono_triangle(bits):
    """bits: tuple of 6 values in {0,1}, one per edge in EDGES order."""
    for idxs in TRIANGLE_EDGE_IDX:
        vals = [bits[i] for i in idxs]
        if vals[0] == vals[1] == vals[2]:
            return True
    return False


def bits_to_int(bits):
    # Qiskit little-endian convention: qubit 0 is least-significant bit.
    val = 0
    for i, b in enumerate(bits):
        val |= (b << i)
    return val


N = 6
DIM = 2 ** N  # 64

good_states = []  # integers (little-endian) with no monochromatic triangle
for bits in product([0, 1], repeat=N):
    if not has_mono_triangle(bits):
        good_states.append(bits_to_int(bits))

good_states = sorted(set(good_states))
CLASSICAL_GOOD_COUNT = len(good_states)

print(f"K4 has {len(EDGES)} edges, {len(TRIANGLES)} triangles, "
      f"{DIM} total 2-colorings.")
print(f"Classical brute force: {CLASSICAL_GOOD_COUNT} colorings avoid a "
      f"monochromatic triangle out of {DIM} total (existence of >0 such "
      f"colorings is consistent with R(3,3)=6, i.e. K4 is below the "
      f"Ramsey threshold).")

assert CLASSICAL_GOOD_COUNT > 0, (
    "Expected at least one triangle-free 2-coloring of K4 (R(3,3)=6 > 4)."
)

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle as a phase-flip diagonal derived directly from
#    the classical predicate above (not copied from any table).
# ---------------------------------------------------------------------------

diag = np.ones(DIM, dtype=complex)
for s in good_states:
    diag[s] = -1.0

oracle = Diagonal(diag.tolist())  # genuine phase oracle over all 6 qubits

grover_op = GroverOperator(oracle=oracle)

# Optimal number of Grover iterations for M marked out of N states:
# floor( pi/4 * sqrt(N/M) )
M = CLASSICAL_GOOD_COUNT
iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(DIM / M))))
print(f"Using {iterations} Grover iteration(s) for M={M}, N={DIM}.")

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.append(grover_op.to_instruction(), range(N))
qc.measure(range(N), range(N))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
SHOTS = 4096
result = backend.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical-register bitstrings are printed MSB(qubit N-1) ... LSB(qubit 0),
# matching our little-endian bits_to_int convention when read as an integer.
measured_ints = {int(bitstr, 2): c for bitstr, c in counts.items()}

good_mass = sum(c for s, c in measured_ints.items() if s in good_states)
good_prob = good_mass / SHOTS
uniform_baseline = M / DIM

top_state = max(measured_ints.items(), key=lambda kv: kv[1])[0]
top_bits = tuple((top_state >> i) & 1 for i in range(N))
top_is_good = not has_mono_triangle(top_bits)

print(f"Measured probability mass on 'good' (no mono triangle) colorings: "
      f"{good_prob:.3f} (uniform baseline would be {uniform_baseline:.3f})")
print(f"Most frequent measured coloring: bits={top_bits} "
      f"(edges {EDGES}), no-mono-triangle={top_is_good}")

# ---------------------------------------------------------------------------
# 4. PASS/FAIL: Grover search must find a genuinely valid witness, and must
#    amplify the good subspace well above the uniform baseline.
# ---------------------------------------------------------------------------

ok_witness = top_is_good
ok_amplification = good_prob > 2 * uniform_baseline  # clear amplification

if ok_witness and ok_amplification and CLASSICAL_GOOD_COUNT > 0:
    print("PASS")
else:
    print("FAIL")
