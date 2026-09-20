"""
Erdos problem #562 -- quantum-testable lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: 562"):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["possible"]
    tags: ["graph theory", "ramsey theory", "hypergraphs"]

LIMITATION, stated honestly up front: problem 562's yaml entry carries no real
OEIS sequence id -- the field literally contains the placeholder string
"possible", not a lookup-able A-number. There is therefore no OEIS sequence to
derive a property from for this problem, and this script does NOT use an OEIS
id. What follows is the best-honest substitute allowed by the task instructions
for this case: a small, finite, genuinely computable property drawn directly
from problem 562's own tags (graph theory / Ramsey theory), checked classically
from first principles and then verified with a real Grover search circuit.

Chosen property
----------------
Take the complete graph K4 (6 edges, 4 triangles). A 2-coloring of the edges
(color 0 / color 1) is "triangle-free" if none of the 4 triangles of K4 is
monochromatic. This is exactly a tiny instance of the classical 2-color Ramsey
question "does every 2-coloring of K_n contain a monochromatic triangle?"
(R(3,3) = 6, so for n = 4 such colorings are known to exist -- but we do not
just cite that fact, we enumerate and check it below).

Search space: all 2^6 = 64 edge-colorings of K4, encoded as 6-bit strings
(one bit per edge). Property: bitstring x is "good" iff none of K4's 4
triangles is monochromatic under coloring x.

Classical step (first principles, done in this script):
    - Enumerate all 64 colorings.
    - For each, check its 4 triangles for monochromaticity.
    - Record the exact set of good colorings.

Quantum step:
    - Build a 6-qubit Grover search whose oracle marks exactly the good
      colorings (the oracle's marked set is the classically-computed set
      above -- not an arbitrary guess, and not copied from OEIS).
    - Run the optimal number of Grover iterations for this search space, on
      the ideal AerSimulator.
    - Sample the resulting distribution and take the most likely outcome.

PASS/FAIL: the script prints PASS iff the most-frequent Grover measurement
outcome is genuinely a "good" (triangle-free) coloring, as independently
re-checked by the classical triangle test -- i.e. the quantum search result
is verified against the classical answer, not asserted.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

# K4 vertices 0,1,2,3. Edges in a fixed order -> bit index.
VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles
assert len(TRIANGLES) == 4


def triangle_edges(tri):
    a, b, c = tri
    return [EDGE_INDEX[tuple(sorted((a, b)))],
            EDGE_INDEX[tuple(sorted((b, c)))],
            EDGE_INDEX[tuple(sorted((a, c)))]]


TRIANGLE_EDGE_IDS = [triangle_edges(t) for t in TRIANGLES]


def is_good_coloring(bits):
    """bits: tuple of 6 ints (0/1), one per edge in EDGES order.
    Returns True iff no triangle of K4 is monochromatic under this coloring."""
    for tri_edges in TRIANGLE_EDGE_IDS:
        colors = {bits[i] for i in tri_edges}
        if len(colors) == 1:
            return False  # monochromatic triangle found
    return True


def bits_from_int(n, width=6):
    return tuple((n >> i) & 1 for i in range(width))


# Enumerate the full classical search space (2^6 = 64 colorings).
GOOD_INTS = [n for n in range(64) if is_good_coloring(bits_from_int(n))]

assert len(GOOD_INTS) > 0, "expected at least one triangle-free coloring of K4"
print(f"Classical enumeration: {len(GOOD_INTS)} / 64 colorings of K4's edges "
      f"avoid a monochromatic triangle.")
print(f"Example good coloring (int): {GOOD_INTS[0]} -> bits {bits_from_int(GOOD_INTS[0])}")

# ---------------------------------------------------------------------------
# 2. Grover search circuit whose oracle marks exactly GOOD_INTS.
# ---------------------------------------------------------------------------

N_QUBITS = 6
N_STATES = 2 ** N_QUBITS


def build_oracle(marked_ints, n_qubits):
    """Phase oracle: flips the sign of every basis state whose integer value
    is in marked_ints. Implemented with X-gates + a multi-controlled Z for
    each marked pattern (n_qubits is small, so this is fully explicit)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked_ints:
        bits = bits_from_int(m, n_qubits)
        # Flip qubits that should be 0 so the target pattern becomes all-1s.
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
        # Multi-controlled Z on all n_qubits (phase flip when all controls=1).
        if n_qubits == 1:
            qc.z(0)
        else:
            mcz = MCXGate(n_qubits - 1)
            qc.h(n_qubits - 1)
            qc.append(mcz, list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        mcz = MCXGate(n_qubits - 1)
        qc.append(mcz, list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(GOOD_INTS, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

# Optimal number of Grover iterations for M marked items out of N states.
M = len(GOOD_INTS)
theta = math.asin(math.sqrt(M / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N_QUBITS))
    qc.append(diffuser.to_instruction(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical register string is little-endian in bit order (c[0] is
# rightmost). Convert each outcome back to our integer/bit convention.
def outcome_str_to_int(bitstr):
    # bitstr as printed by Qiskit: c[n-1] ... c[1] c[0]
    rev = bitstr[::-1]  # now rev[i] == c[i] == qubit i's measured bit
    return int(rev, 2)

best_bitstr = max(counts, key=counts.get)
best_int = outcome_str_to_int(best_bitstr)
best_prob = counts[best_bitstr] / shots

print(f"Grover ran {iterations} iteration(s) over {N_QUBITS} qubits "
      f"({N_STATES} states, {M} marked).")
print(f"Most frequent measurement: {best_bitstr} -> coloring int {best_int} "
      f"(probability {best_prob:.3f} over {shots} shots)")

# ---------------------------------------------------------------------------
# 4. Verify the quantum result against the classical answer.
# ---------------------------------------------------------------------------

quantum_claims_good = best_int in GOOD_INTS
classically_good = is_good_coloring(bits_from_int(best_int))
# Sanity: these two checks must agree by construction (GOOD_INTS was built
# from is_good_coloring in the first place); re-derive independently anyway.
assert quantum_claims_good == classically_good

# With M = 18 marked states out of 64, Grover amplifies the *total*
# probability mass on the marked subspace, which is then spread roughly
# evenly across those 18 states -- so no single outcome need exceed 50% on
# its own. Verify (a) the single most-likely outcome is itself a genuine,
# classically re-checked triangle-free coloring, and (b) the total measured
# probability mass landing on ANY triangle-free coloring is amplified well
# above the pre-Grover uniform baseline of M/64 = 0.281.
total_good_prob = sum(
    c for bitstr, c in counts.items() if outcome_str_to_int(bitstr) in GOOD_INTS
) / shots
baseline_prob = M / N_STATES
print(f"Total probability mass on triangle-free colorings: {total_good_prob:.3f} "
      f"(pre-Grover uniform baseline: {baseline_prob:.3f})")

verified = classically_good and total_good_prob > 0.9

if verified:
    print("PASS: Grover search on K4's 6 edge-coloring qubits returned a "
          "triangle-free 2-coloring with high probability, matching the "
          "classical enumeration.")
else:
    print("FAIL: Grover search did not converge on a classically-verified "
          "triangle-free coloring with sufficient confidence.")

assert verified, "quantum result did not match the classical answer with sufficient confidence"
