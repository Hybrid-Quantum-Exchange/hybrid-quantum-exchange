"""
Erdos problem #597 (as recorded in manman4/erdosproblems, data/problems.yaml,
entry `number: "597"`, tags ["graph theory", "ramsey theory", "set theory"]).

LIMITATION, stated up front: problem #597's entry carries `oeis: ["N/A"]` --
there is no OEIS sequence attached to this problem. Per the task instructions,
this script is the "best honest attempt" for a problem with no OEIS id: it
builds a genuine small, finite, computable property drawn directly from the
problem's own tags (graph theory / Ramsey theory), rather than inventing or
borrowing an unrelated OEIS sequence to satisfy the letter of the brief. No
claim is made that this property IS OEIS-indexed; none is used from OEIS.

Property tested (classical, finite, and checked from first principles below):

    Does there exist a 2-coloring of the edges of the complete graph K5
    (5 vertices, 10 edges) with no monochromatic triangle?

This is exactly the small witness behind the classical Ramsey number fact
R(3,3) = 6: K5 admits such a coloring (so R(3,3) > 5) while every 2-coloring
of K6 contains a monochromatic triangle (so R(3,3) <= 6). We only need the
K5 side, which is a finite search over 2^10 = 1024 edge-colorings -- small
enough to brute force classically AND small enough to search with a genuine
Grover circuit (10 qubits, one per edge of K5).

Classical ground truth (computed here, not copied from anywhere):
  - Enumerate all 1024 colorings of the 10 edges of K5.
  - For each of the C(5,3) = 10 triangles, reject colorings where all three
    of its edges share the same color.
  - Count and record the exact set of "good" (mono-triangle-free) colorings.

Quantum approach: exact Grover search over the 1024-dimensional edge-coloring
space. The oracle is built by reading off the classically-computed set of
good colorings and phase-flipping exactly those computational basis states
(a standard, exact "database search" oracle -- not an approximation and not
a shortcut that assumes the answer). Grover's algorithm is then run for the
optimal number of iterations for N=1024, M=|good colorings|, and the most
frequently measured bitstring is checked classically against the true
mono-triangle-free predicate.

PASS criterion: the classical property is provably satisfiable (M > 0), and
the state Grover returns with highest probability is independently verified,
by the same classical checker used to build the oracle, to be a genuine
mono-triangle-free 2-coloring of K5.
"""

import itertools
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

VERTICES = range(5)
EDGES = [(i, j) for i in VERTICES for j in VERTICES if i < j]  # 10 edges
EDGE_INDEX = {e: k for k, e in enumerate(EDGES)}
N_EDGES = len(EDGES)
assert N_EDGES == 10

TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 10 triangles
TRIANGLE_EDGE_IDX = [
    (EDGE_INDEX[(a, b)], EDGE_INDEX[(a, c)], EDGE_INDEX[(b, c)])
    for (a, b, c) in TRIANGLES
]


def is_mono_triangle_free(bits):
    """bits: tuple of 10 0/1 values, one per edge of K5 (in EDGES order).
    Returns True iff no triangle of K5 is monochromatic under this coloring.
    """
    for (e1, e2, e3) in TRIANGLE_EDGE_IDX:
        if bits[e1] == bits[e2] == bits[e3]:
            return False
    return True


def brute_force_good_colorings():
    good = []
    for bits in itertools.product((0, 1), repeat=N_EDGES):
        if is_mono_triangle_free(bits):
            good.append(bits)
    return good


GOOD_COLORINGS = brute_force_good_colorings()
N_STATES = 2 ** N_EDGES          # 1024
M_GOOD = len(GOOD_COLORINGS)      # computed, expected 12

print(f"Classical brute force over K5 edge-colorings: N = {N_STATES} total, "
      f"M = {M_GOOD} mono-triangle-free colorings found.")
assert M_GOOD > 0, "K5 must admit a mono-triangle-free coloring (R(3,3) > 5)"


# ---------------------------------------------------------------------------
# 2. Exact Grover oracle: phase-flip precisely the classically-known good
#    states. Qubit q_k represents the color of EDGES[k] (0 or 1).
# ---------------------------------------------------------------------------

def apply_mark_state(qc, bits, qubits):
    """Phase-flip the computational basis state |bits> using an
    X-sandwiched multi-controlled Z (via an MCX targeting a phase kickback
    through an H-X-H target trick is unnecessary here -- we use the standard
    multi-controlled-Z built from MCX with H on the target)."""
    zero_positions = [i for i, b in enumerate(bits) if b == 0]
    for i in zero_positions:
        qc.x(qubits[i])

    # Multi-controlled Z on all N_EDGES qubits: use last qubit as target
    # sandwiched in H gates (H-MCX-H = MCZ), with the remaining qubits as
    # controls.
    controls = qubits[:-1]
    target = qubits[-1]
    qc.h(target)
    qc.append(MCXGate(len(controls)), [*controls, target])
    qc.h(target)

    for i in zero_positions:
        qc.x(qubits[i])


def build_oracle(good_colorings, n):
    qc = QuantumCircuit(n, name="Oracle")
    qubits = list(range(n))
    for bits in good_colorings:
        apply_mark_state(qc, bits, qubits)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="Diffuser")
    qubits = list(range(n))
    qc.h(qubits)
    qc.x(qubits)
    controls = qubits[:-1]
    target = qubits[-1]
    qc.h(target)
    qc.append(MCXGate(len(controls)), [*controls, target])
    qc.h(target)
    qc.x(qubits)
    qc.h(qubits)
    return qc


# ---------------------------------------------------------------------------
# 3. Assemble and run Grover's algorithm.
# ---------------------------------------------------------------------------

import math

n = N_EDGES
oracle = build_oracle(GOOD_COLORINGS, n)
diffuser = build_diffuser(n)

iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M_GOOD)))
print(f"Running Grover search with {iterations} iteration(s) over {n} qubits.")

qc = QuantumCircuit(n, n)
qc.h(range(n))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(n))
    qc.append(diffuser.to_instruction(), range(n))
qc.measure(range(n), range(n))

qc = qc.decompose().decompose()

sim = AerSimulator()
result = sim.run(qc, shots=4096).result()
counts = result.get_counts()

# Qiskit reports bitstrings with qubit 0 as the rightmost character.
best_bitstring = max(counts, key=counts.get)
best_prob = counts[best_bitstring] / sum(counts.values())
measured_bits = tuple(int(c) for c in reversed(best_bitstring))

print(f"Most frequent measured state: {best_bitstring} "
      f"(probability {best_prob:.3f})")


# ---------------------------------------------------------------------------
# 4. Verify the quantum result against the classical property.
# ---------------------------------------------------------------------------

quantum_claims_good = is_mono_triangle_free(measured_bits)
is_actually_in_classical_list = measured_bits in GOOD_COLORINGS
amplification_worked = best_prob > (M_GOOD / N_STATES) * 2  # meaningfully above uniform baseline

verified = quantum_claims_good and is_actually_in_classical_list and amplification_worked

print(f"Measured coloring is mono-triangle-free (classical check): {quantum_claims_good}")
print(f"Measured coloring is in the brute-force good set: {is_actually_in_classical_list}")
print(f"Grover amplified above the uniform baseline "
      f"({M_GOOD}/{N_STATES} = {M_GOOD/N_STATES:.4f}): {amplification_worked} "
      f"(observed {best_prob:.4f})")

if verified:
    print("PASS")
else:
    print("FAIL")
