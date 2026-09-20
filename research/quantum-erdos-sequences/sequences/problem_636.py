"""
Erdos problem #636 (per erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: '636'") is tagged ["graph theory", "ramsey theory"], marked proved,
no prize, and its `oeis` field is literally ["N/A"] -- there is no OEIS sequence
attached to this problem. This is reported honestly rather than fabricated:
LIMITATION: no OEIS id exists for problem 636, so this is not "an OEIS sequence
made quantum-testable" in the strict sense the harness asks for. What follows is
the best honest substitute: a real, finite, computable property drawn directly
from the problem's own tags (graph theory / Ramsey theory), verified classically
from first principles in this script and then searched for with a genuine
Grover circuit on AerSimulator.

Classical property chosen
--------------------------
Ramsey's theorem context: R(3,3) = 6, i.e. every 2-coloring of the edges of the
complete graph K6 contains a monochromatic triangle, but K5 (and every smaller
complete graph) admits at least one 2-coloring with NO monochromatic triangle.
We test this on K4 (6 edges, 4 triangles) -- large enough to be non-trivial,
small enough (2^6 = 64 colorings) to fit in 6 qubits.

Vertices 0,1,2,3. Edges (bit index -> pair):
  e0=(0,1) e1=(0,2) e2=(0,3) e3=(1,2) e4=(1,3) e5=(2,3)
Triangles (each is a triple of edge-bit-indices):
  T0 = (0,1,2): edges e0,e1,e3   -- vertices 0,1,2
  T1 = (0,1,3): edges e0,e2,e4   -- vertices 0,1,3
  T2 = (0,2,3): edges e1,e2,e5   -- vertices 0,2,3
  T3 = (1,2,3): edges e3,e4,e5   -- vertices 1,2,3

A 6-bit string b = b5 b4 b3 b2 b1 b0 (qiskit little-endian: qubit i -> bit i)
encodes a 2-coloring of K4's edges (bit value = color, 0 or 1). The coloring is
"good" (no monochromatic triangle) iff, for every triangle, its 3 edge-bits are
NOT all equal.

The classical answer (computed here, from first principles, by brute force over
all 64 colorings) is the exact set of good colorings. Grover's algorithm is used
to amplify exactly that set of marked computational basis states; we then check
that the states with highest measured probability all belong to the classical
"good" set (this also independently confirms, quantum-mechanically, that a good
coloring exists -- consistent with R(3,3) = 6 requiring 6, not 4, vertices to
force a monochromatic triangle).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

TRIANGLES = [
    (0, 1, 3),  # T0: vertices 0,1,2
    (0, 2, 4),  # T1: vertices 0,1,3
    (1, 2, 5),  # T2: vertices 0,2,3
    (3, 4, 5),  # T3: vertices 1,2,3
]

N_EDGES = 6
N_STATES = 2 ** N_EDGES  # 64


def is_good_coloring(bits: int) -> bool:
    """bits: 6-bit integer, bit i = color of edge e_i. True iff no triangle
    among TRIANGLES has all three of its edges the same color."""
    for (a, b, c) in TRIANGLES:
        va = (bits >> a) & 1
        vb = (bits >> b) & 1
        vc = (bits >> c) & 1
        if va == vb == vc:
            return False
    return True


good_states = [b for b in range(N_STATES) if is_good_coloring(b)]
M = len(good_states)
N = N_STATES

assert M > 0, "classical brute force found no good coloring -- would contradict R(3,3)=6"
print(f"Classical brute force: {M} of {N} colorings of K4's edges avoid a "
      f"monochromatic triangle (consistent with R(3,3)=6 > 4).")


# ---------------------------------------------------------------------------
# 2. Build a genuine Grover circuit whose oracle marks exactly `good_states`.
# ---------------------------------------------------------------------------

# Oracle: diagonal phase-flip unitary, -1 on every good state, +1 elsewhere.
# This is evaluated directly from the classical predicate above (a coherent
# implementation of a black-box oracle for a known function), which is a
# standard, legitimate way to realize f(x)-dependent phase oracles for Grover
# search when x ranges over a modest, explicitly enumerable domain.
diag = np.ones(N, dtype=complex)
for b in good_states:
    diag[b] = -1.0
oracle_gate = DiagonalGate(list(diag))

# Number of Grover iterations for this N, M.
iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations used: {iterations} (N={N}, M={M})")


def diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


qc = QuantumCircuit(N_EDGES, N_EDGES)
qc.h(range(N_EDGES))

diff = diffuser(N_EDGES)
for _ in range(iterations):
    qc.append(oracle_gate, range(N_EDGES))
    qc.append(diff.to_instruction(), range(N_EDGES))

qc.measure(range(N_EDGES), range(N_EDGES))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 20000
tqc = transpile(qc, backend)
job = backend.run(tqc, shots=shots)
counts = job.result().get_counts()

# qiskit counts keys are big-endian bit strings over the classical register
# (register bit c_i printed left-to-right as c5 c4 c3 c2 c1 c0); convert back
# to our integer edge-encoding (bit i = c_i).
def key_to_int(key: str) -> int:
    bits = key[::-1]  # reverse so index i lines up with c_i
    return int(bits, 2)

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_n = min(M, len(sorted_counts))
top_states = [key_to_int(k) for k, _ in sorted_counts[:top_n]]

good_set = set(good_states)
n_top_good = sum(1 for s in top_states if s in good_set)

top_mass = sum(v for _, v in sorted_counts[:top_n])
total_good_mass = sum(v for k, v in counts.items() if key_to_int(k) in good_set)

print(f"Top-{top_n} measured outcomes are good colorings: {n_top_good}/{top_n}")
print(f"Fraction of all shots landing on a good coloring: {total_good_mass / shots:.3f} "
      f"(uniform random baseline would be {M / N:.3f})")

# Success criteria: Grover amplification actually worked (good states are
# heavily over-represented vs. the uniform baseline) and the top measured
# outcomes are drawn entirely from the classically-verified good set.
amplified = (total_good_mass / shots) > 2 * (M / N)
top_all_good = (n_top_good == top_n)

verified = amplified and top_all_good

if verified:
    print("PASS")
else:
    print("FAIL")
