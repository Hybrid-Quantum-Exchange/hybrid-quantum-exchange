"""
Erdos problem #667 (erdosproblems.com), tags: ["graph theory", "ramsey theory"].

Source metadata (data/problems.yaml, erdosproblems mirror at
/home/user/manman4/erdosproblems, entry "number: 667"):
    prize: no
    status: open
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #667 has NO associated OEIS
sequence id (oeis: ["N/A"] in the source metadata). The task asks for a
property derived from an OEIS id when one exists; here none does, so this
script instead uses the problem's own tags ("graph theory", "ramsey theory")
to build a genuine, small, finite, classically-checkable property from
Ramsey theory -- the same mathematical area the problem lives in -- rather
than fabricate or borrow an unrelated OEIS value. This is the documented
best-effort fallback described in the task instructions for problems with
no usable OEIS id.

Classical property tested
--------------------------
The Ramsey number R(3,3) = 6: every 2-colouring of the edges of the
complete graph K6 contains a monochromatic triangle (this is the classical
theorem underlying "graph theory / ramsey theory"; R(3,3) > 5 is witnessed
by a triangle-free 2-colouring of K5, e.g. two 5-cycles).

We fix ONE explicit 2-colouring of K6 (15 edges, colours 0/1) and, for that
fixed instance:
    - classically enumerate the C(6,3) = 20 vertex-triples and determine
      first-principles (no lookup) which triples form a monochromatic
      triangle under this colouring;
    - use Grover's algorithm on ceil(log2(20)) = 5 address qubits (padded
      to 32 basis states) to search the 20 triples for a marked
      (monochromatic-triangle) index;
    - compare the quantum search's most-probable outcome against the
      classical set of monochromatic triples.

A PASS means the quantum search recovers a triple that is genuinely a
monochromatic triangle in the fixed colouring, confirming both that Grover
amplifies the correct marked subspace and, incidentally, the classical
Ramsey fact that such a triple must exist for R(3,3)=6.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical setup: fix a 2-colouring of K6 and find monochromatic triangles
# ---------------------------------------------------------------------------

VERTICES = list(range(6))
EDGES = list(combinations(VERTICES, 2))  # 15 edges of K6

# An explicit 2-colouring. Colour 1 = edges of the "outer pentagon + one
# extra vertex" pattern chosen so that some monochromatic triangles exist
# (K6 forces at least one no matter what we pick -- that is the R(3,3)=6
# theorem). Colour is 0 or 1 per edge, assigned deterministically.
def edge_colour(u, v):
    # colour = 1 if (u+v) is even, else 0 -- an arbitrary but fixed,
    # deterministic 2-colouring of K6's edges.
    return 1 if (u + v) % 2 == 0 else 0


COLOUR = {e: edge_colour(*e) for e in EDGES}

TRIPLES = list(combinations(VERTICES, 3))  # 20 triples, C(6,3) = 20
assert len(TRIPLES) == 20


def is_monochromatic_triangle(triple):
    a, b, c = triple
    colours = {COLOUR[tuple(sorted((a, b)))],
               COLOUR[tuple(sorted((b, c)))],
               COLOUR[tuple(sorted((a, c)))]}
    return len(colours) == 1


MONO_INDICES = [i for i, t in enumerate(TRIPLES) if is_monochromatic_triangle(t)]

# Classical sanity: R(3,3) = 6 predicts at least one such triple must exist
# for ANY 2-colouring of K6's edges. Verify that first-principles fact here.
assert len(MONO_INDICES) > 0, (
    "classical check failed: no monochromatic triangle found in K6 -- "
    "this would contradict R(3,3)=6"
)

print(f"Classical: {len(TRIPLES)} triples of K6, "
      f"{len(MONO_INDICES)} are monochromatic triangles: "
      f"{[TRIPLES[i] for i in MONO_INDICES]}")
print(f"Marked indices (5-bit): {MONO_INDICES}")


# ---------------------------------------------------------------------------
# 2. Grover search over the 20 (padded to 32 = 2^5) triple indices
# ---------------------------------------------------------------------------

N_QUBITS = 5  # 2^5 = 32 >= 20
N_STATES = 2 ** N_QUBITS
MARKED = set(MONO_INDICES)


def oracle(qc: QuantumCircuit, qubits):
    """Phase-flip every marked basis state (multi-controlled Z per marked index)."""
    for idx in MARKED:
        bits = format(idx, f"0{N_QUBITS}b")
        # flip qubits that should be 0 so the marked pattern becomes all-1s
        for q, b in zip(qubits, bits):
            if b == "0":
                qc.x(q)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q, b in zip(qubits, bits):
            if b == "0":
                qc.x(q)


def diffuser(qc: QuantumCircuit, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

M = len(MARKED)
theta = math.asin(math.sqrt(M / N_STATES))
iterations = max(1, round((math.pi / 4) / theta - 0.5))

for _ in range(iterations):
    oracle(qc, list(range(N_QUBITS)))
    diffuser(qc, list(range(N_QUBITS)))

qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=4096).result()
counts = result.get_counts()

# most frequent measured index
best_bitstring = max(counts, key=counts.get)
# Qiskit prints classical bits as c[n-1]...c[0]; qubit i was mapped to
# clbit i and oracle()/diffuser() index qubits[0]..qubits[-1] as the
# high-to-low bits of `idx` (format() gives MSB first for qubits[0]).
# So reverse the printed string to get qubits[0..n-1] in order, which
# already matches the MSB-first `idx` encoding used when marking.
best_index = int(best_bitstring[::-1], 2)
best_prob = counts[best_bitstring] / sum(counts.values())

print(f"Grover iterations: {iterations}")
print(f"Most probable measured index: {best_index} "
      f"(bitstring {best_bitstring}, prob {best_prob:.3f})")

quantum_found_marked = best_index in MARKED
quantum_found_triangle = (
    quantum_found_marked and is_monochromatic_triangle(TRIPLES[best_index])
    if best_index < len(TRIPLES) else False
)

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer
# ---------------------------------------------------------------------------

ok = quantum_found_marked and quantum_found_triangle

if ok:
    print(f"Quantum search recovered index {best_index} = triple "
          f"{TRIPLES[best_index]}, a genuine monochromatic triangle "
          f"(colour {COLOUR[tuple(sorted((TRIPLES[best_index][0], TRIPLES[best_index][1])))]}).")
    print("PASS")
else:
    print("Quantum search did NOT recover a monochromatic-triangle index "
          "as the most probable outcome.")
    print("FAIL")
