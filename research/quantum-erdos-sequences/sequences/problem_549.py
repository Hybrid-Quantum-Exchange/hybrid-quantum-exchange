"""
Erdos problem #549 -- quantum-testable instance.

Source metadata (from erdosproblems data/problems.yaml, entry "number: 549"):
    prize: no
    informal_status: disproved (last_update 2025-08-31)
    oeis: ["N/A"]          <-- no OEIS sequence id is attached to this problem
    tags: ["graph theory", "ramsey theory"]

LIMITATION, stated honestly up front: problem #549 carries no OEIS id, so
there is no OEIS sequence to build a membership/term-defining oracle from
for it directly, and the required "identify a property from its OEIS
sequence id(s)" step cannot literally be done. Per the task's fallback
instructions, this script instead builds its best honest attempt at a
genuine, finite, computable property rooted in the problem's own tags
(graph theory / Ramsey theory), rather than fabricating or mis-attributing
an OEIS id that problem 549 does not have.

Chosen property (real math, not fabricated):
    Let K4 be the complete graph on 4 vertices (vertex set {0,1,2,3}), which
    has C(4,2) = 6 edges and exactly C(4,3) = 4 triangles. Consider all
    2^6 = 64 ways to 2-colour the edges of K4 with colours {0,1} ("red"/
    "blue"). This is the smallest nontrivial instance of the 2-colour
    Ramsey question R(3,3): does every 2-colouring of a complete graph on n
    vertices contain a monochromatic triangle? It is classically known that
    R(3,3) = 6, so for n = 4 (< 6) there MUST exist at least one 2-colouring
    of K4 with NO monochromatic triangle.

    The finite, computable property tested here: "the set S of edge
    2-colourings of K4 that contain no monochromatic triangle is nonempty,
    and Grover search over the 64 colourings finds an element of S with
    high probability." |S| is computed from first principles below by exact
    brute force over all 64 colourings (no literature value is copied), and
    that same brute-force set is used to build the Grover oracle and to
    grade the quantum result.

Circuit: 6 qubits (one per edge of K4). A classically-computed "good state"
set S (colourings with zero monochromatic triangles among the 4 triangles
of K4) is marked by a diagonal phase-flip oracle (phase -1 on states in S,
+1 elsewhere) followed by the standard Grover diffusion operator, iterated
the optimal (rounded) number of times for a search space of size 64 and
|S| solutions. The circuit is run on the ideal AerSimulator (statevector
method) and the final measurement distribution's probability mass on S is
checked against the theoretical Grover amplification bound and against the
classical count |S|.

PASS criterion: |S| > 0 (matching the classical R(3,3) = 6 fact) AND the
quantum circuit's measured probability of landing in S exceeds a
generous-but-meaningful threshold, confirming the oracle+Grover circuit
actually amplifies the classically-defined good set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator

N_VERTICES = 4
VERTICES = list(range(N_VERTICES))
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges, fixed order
N_EDGES = len(EDGES)
assert N_EDGES == 6

TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles
assert len(TRIANGLES) == 4

EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}


def triangle_edge_bits(colouring_bits, triangle):
    """Return the 3 colour bits (0/1) of a triangle's edges under a colouring."""
    bits = []
    for a, b in itertools.combinations(triangle, 2):
        e = (a, b) if (a, b) in EDGE_INDEX else (b, a)
        bits.append(colouring_bits[EDGE_INDEX[e]])
    return bits


def is_monochromatic(bits3):
    return bits3[0] == bits3[1] == bits3[2]


def colouring_has_no_mono_triangle(colouring_bits):
    for tri in TRIANGLES:
        if is_monochromatic(triangle_edge_bits(colouring_bits, tri)):
            return False
    return True


# ---------------------------------------------------------------------------
# Step 1: classical, first-principles brute force over all 2^6 = 64 colourings
# ---------------------------------------------------------------------------

good_states = []  # list of integers 0..63 whose binary expansion has no mono triangle
for state in range(2 ** N_EDGES):
    bits = [(state >> i) & 1 for i in range(N_EDGES)]
    if colouring_has_no_mono_triangle(bits):
        good_states.append(state)

CLASSICAL_GOOD_COUNT = len(good_states)
CLASSICAL_GOOD_SET = set(good_states)

print(f"Classical brute force over all {2 ** N_EDGES} edge 2-colourings of K4:")
print(f"  colourings with NO monochromatic triangle: {CLASSICAL_GOOD_COUNT}")
print(f"  (this must be > 0, since R(3,3) = 6 > 4 vertices)")
assert CLASSICAL_GOOD_COUNT > 0, "classical fact R(3,3)=6 implies this must be nonempty"

# ---------------------------------------------------------------------------
# Step 2: build a Grover oracle that phase-flips exactly the classically
# computed good_states, then amplify with the standard diffusion operator.
# ---------------------------------------------------------------------------

n = N_EDGES  # 6 qubits, search space size N = 64
N = 2 ** n
M = CLASSICAL_GOOD_COUNT

diag = np.ones(N, dtype=complex)
for s in good_states:
    diag[s] = -1.0


def oracle_gate():
    qc = QuantumCircuit(n, name="Oracle")
    qc.append(DiagonalGate(list(diag)), list(range(n)))
    return qc.to_gate()


def diffusion_gate():
    qc = QuantumCircuit(n, name="Diffusion")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc.to_gate()


# optimal number of Grover iterations for N states, M marked
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(n, n)
qc.h(range(n))
oracle = oracle_gate()
diffuser = diffusion_gate()
for _ in range(iterations):
    qc.append(oracle, range(n))
    qc.append(diffuser, range(n))
qc.measure(range(n), range(n))

print(f"\nGrover search: N={N} states, M={M} marked (no-mono-triangle colourings), "
      f"iterations={iterations}")

# ---------------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator(method="statevector")
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical register bitstring is c[n-1]...c[0]; reverse to match
# our little-endian edge index convention (qubit i <-> EDGES[i]).
good_shots = 0
for bitstring, n_shots in counts.items():
    state_int = int(bitstring[::-1], 2)
    if state_int in CLASSICAL_GOOD_SET:
        good_shots += n_shots

measured_prob_good = good_shots / shots
theoretical_prob_good = math.sin((2 * iterations + 1) * theta) ** 2

print(f"Measured probability of landing on a no-mono-triangle colouring: "
      f"{measured_prob_good:.4f}")
print(f"Theoretical Grover amplitude-amplification bound: "
      f"{theoretical_prob_good:.4f}")

# ---------------------------------------------------------------------------
# Step 4: verify quantum result against the classical computation
# ---------------------------------------------------------------------------

THRESHOLD = 0.85  # generous but meaningful: random guessing gives M/N
random_baseline = M / N

verified = (
    CLASSICAL_GOOD_COUNT > 0
    and measured_prob_good >= THRESHOLD
    and measured_prob_good > 3 * random_baseline
)

print(f"\nRandom-guessing baseline probability: {random_baseline:.4f}")
print(f"Verification threshold: {THRESHOLD}")

if verified:
    print("\nPASS: Grover search on the ideal AerSimulator amplifies the "
          "classically-verified set of K4 edge-colourings with no "
          "monochromatic triangle (consistent with R(3,3)=6), matching the "
          "first-principles classical computation.")
else:
    print("\nFAIL: quantum result did not match the classical computation "
          "within the verification threshold.")
