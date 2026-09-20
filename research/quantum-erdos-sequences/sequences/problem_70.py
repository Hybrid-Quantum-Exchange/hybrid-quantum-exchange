"""
Erdos problem #70 -- quantum-testable instance.

Source metadata (data/problems.yaml, manman4/erdosproblems, read-only clone):
  number: 70
  tags: ["graph theory", "ramsey theory", "set theory"]
  oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #70's YAML entry carries no
OEIS sequence id ("N/A"). There is therefore no actual integer sequence to
build a "membership" or "early term" oracle from for this problem, and this
script cannot honestly claim to test an OEIS sequence for problem 70. What
it does instead, as the best faithful attempt available, is take the one
piece of real mathematical content in the entry -- the "ramsey theory" /
"graph theory" tags -- and build a genuine, finite, classically-checkable
property from Ramsey-type combinatorics that a small quantum circuit can
search: monochromatic-triangle-free 2-colorings of the edges of the
complete graph K4.

Classical property under test
------------------------------
Let K4 have vertices {0,1,2,3} and its 6 edges
    e0=(0,1) e1=(0,2) e2=(0,3) e3=(1,2) e4=(1,3) e5=(2,3)
A 2-coloring of the edges is a bitstring of length 6 (bit i = color of
edge i, 0 or 1). The coloring is "triangle-clean" if none of the 4
triangles of K4 -- (0,1,2), (0,1,3), (0,2,3), (1,2,3) -- is monochromatic
(all three of its edges the same color).

This is exactly the finite combinatorial fact behind the classical Ramsey
number R(3,3)=6: on K6 every 2-coloring contains a monochromatic triangle,
but on K4 (and K5) triangle-clean colorings exist. The script:
  1. Computes, from first principles by brute force over all 2**6 = 64
     edge-colorings of K4, the exact set of triangle-clean colorings (the
     classical answer).
  2. Builds a real Grover search circuit whose phase oracle is computed
     with reversible arithmetic (ancilla-based equality/AND/OR gates
     evaluating the "triangle-clean" predicate on the 6 data qubits, not a
     hardcoded lookup table of the classical answer) and amplifies exactly
     the triangle-clean colorings.
  3. Runs the circuit on the ideal AerSimulator, reads off the most likely
     measured bitstrings, and checks that they are all triangle-clean
     colorings per the classical brute force -- printing PASS or FAIL.

No OEIS id is used because none exists for problem #70; this is disclosed
rather than papered over. ran_ok / verified_against_classical are reported
for this Ramsey-flavored circuit, not for any OEIS sequence.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed by brute force from first principles.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def triangle_edge_indices(tri):
    a, b, c = tri
    pairs = [(a, b), (a, c), (b, c)]
    return [EDGE_INDEX[tuple(sorted(p))] for p in pairs]


TRI_EDGES = [triangle_edge_indices(t) for t in TRIANGLES]


def is_triangle_clean(bits):
    """bits: tuple of 6 ints (0/1), one per edge in EDGES order."""
    for i0, i1, i2 in TRI_EDGES:
        if bits[i0] == bits[i1] == bits[i2]:
            return False
    return True


def classical_solutions():
    sols = []
    for bits in itertools.product([0, 1], repeat=6):
        if is_triangle_clean(bits):
            sols.append(bits)
    return sols


CLASSICAL_SOLUTIONS = classical_solutions()
CLASSICAL_SOLUTION_STRINGS = {
    "".join(str(b) for b in reversed(bits)) for bits in CLASSICAL_SOLUTIONS
    # reversed: Qiskit prints classical bit strings with qubit 0 as the
    # rightmost character, so we store the matching convention.
}

N_DATA = 6
N_SOLUTIONS = len(CLASSICAL_SOLUTIONS)
N_STATES = 2 ** N_DATA

print(f"Classical brute force over {N_STATES} edge-colorings of K4:")
print(f"  triangle-clean colorings found: {N_SOLUTIONS} / {N_STATES}")
assert N_SOLUTIONS > 0, "sanity: K4 must admit triangle-clean colorings (R(3,3)=6)"

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search with a reversible-arithmetic oracle.
# ---------------------------------------------------------------------------
# Qubit layout:
#   data[0..5]   : the 6 edge-color bits (search register)
#   diff[0..1]   : scratch qubits for pairwise XOR within a triangle
#   mono[0..3]   : one ancilla per triangle, set to 1 iff that triangle is
#                  monochromatic (computed reversibly, then uncomputed)
#   bad          : OR of the 4 mono ancillas ("some triangle is mono")
#   out          : phase-kickback target, prepared in |-> ; flipped
#                  (via X on `bad`) exactly when NO triangle is mono, i.e.
#                  exactly on the triangle-clean colorings -- this is the
#                  oracle's phase flip.

DATA = list(range(0, 6))
DIFF = [6, 7]
MONO = [8, 9, 10, 11]
BAD = 12
OUT = 13
N_QUBITS = 14


def build_oracle(qc):
    """Reversible computation of the triangle-clean predicate, with phase
    kickback onto OUT (prepared in |-> by the caller) exactly when the
    current data-register coloring is triangle-clean."""

    # For each triangle, compute mono[i] = 1 iff its 3 edges are equal.
    # equal(a,b,c) <=> (a XOR b == 0) AND (a XOR c == 0)
    for i, (i0, i1, i2) in enumerate(TRI_EDGES):
        d0, d1 = DIFF
        qc.cx(DATA[i0], d0)
        qc.cx(DATA[i1], d0)  # d0 = a XOR b
        qc.cx(DATA[i0], d1)
        qc.cx(DATA[i2], d1)  # d1 = a XOR c
        # mono[i] = NOT d0 AND NOT d1  -> flip both to 1-controls via X
        qc.x(d0)
        qc.x(d1)
        qc.ccx(d0, d1, MONO[i])
        qc.x(d0)
        qc.x(d1)
        # uncompute d0, d1
        qc.cx(DATA[i0], d1)
        qc.cx(DATA[i2], d1)
        qc.cx(DATA[i0], d0)
        qc.cx(DATA[i1], d0)

    # bad = OR(mono[0..3]) ; compute via De Morgan with a multi-controlled
    # X: bad flips to 1 unless ALL mono[i] are 0, so instead build OR
    # directly: bad = NOT(AND of NOT mono[i]).
    for m in MONO:
        qc.x(m)
    qc.mcx(MONO, BAD)  # BAD = AND(NOT mono[i]) so far == NOT(OR mono[i])
    qc.x(BAD)          # BAD = OR(mono[i])  ("some triangle is monochromatic")
    for m in MONO:
        qc.x(m)

    # Flip OUT (which is in |-> ) iff BAD == 0, i.e. iff triangle-clean.
    qc.x(BAD)
    qc.cx(BAD, OUT)
    qc.x(BAD)

    # Uncompute BAD
    for m in MONO:
        qc.x(m)
    qc.mcx(MONO, BAD)
    qc.x(BAD)
    for m in MONO:
        qc.x(m)

    # Uncompute MONO
    for i, (i0, i1, i2) in enumerate(TRI_EDGES):
        d0, d1 = DIFF
        qc.cx(DATA[i0], d0)
        qc.cx(DATA[i1], d0)
        qc.cx(DATA[i0], d1)
        qc.cx(DATA[i2], d1)
        qc.x(d0)
        qc.x(d1)
        qc.ccx(d0, d1, MONO[i])
        qc.x(d0)
        qc.x(d1)
        qc.cx(DATA[i0], d1)
        qc.cx(DATA[i2], d1)
        qc.cx(DATA[i0], d0)
        qc.cx(DATA[i1], d0)


def build_diffuser(qc):
    qc.h(DATA)
    qc.x(DATA)
    qc.h(DATA[-1])
    qc.mcx(DATA[:-1], DATA[-1])
    qc.h(DATA[-1])
    qc.x(DATA)
    qc.h(DATA)


iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / N_SOLUTIONS)))
print(f"Grover iterations chosen: {iterations}")

qc = QuantumCircuit(N_QUBITS, N_DATA)
qc.h(DATA)
qc.x(OUT)
qc.h(OUT)

for _ in range(iterations):
    build_oracle(qc)
    build_diffuser(qc)

qc.measure(DATA, list(range(N_DATA)))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and check against the classical answer.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_k = max(N_SOLUTIONS, 1)
top_results = sorted_counts[:top_k]

print("Top measured bitstrings (qubit0 = rightmost char):")
for bitstring, count in top_results[:10]:
    print(f"  {bitstring}: {count}")

# Verify: the high-probability outcomes should all be triangle-clean
# colorings per the classical brute-force computation above.
total_shots_in_top = sum(c for _, c in top_results)
matched = sum(c for b, c in top_results if b in CLASSICAL_SOLUTION_STRINGS)
match_fraction = matched / total_shots_in_top if total_shots_in_top else 0.0

print(f"Fraction of top-{top_k} outcomes matching classical solutions: "
      f"{match_fraction:.3f}")

# Also sanity check: total probability mass landing on ANY classical
# solution vs. any non-solution, across all shots.
mass_on_solutions = sum(
    c for b, c in counts.items() if b in CLASSICAL_SOLUTION_STRINGS
)
mass_fraction = mass_on_solutions / shots
print(f"Total probability mass on classical solutions (all shots): "
      f"{mass_fraction:.3f}")

verified = match_fraction >= 0.9 and mass_fraction >= 0.6

if verified:
    print("PASS")
else:
    print("FAIL")
