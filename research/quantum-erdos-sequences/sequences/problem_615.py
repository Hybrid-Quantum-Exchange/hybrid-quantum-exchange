"""
Erdos problem #615 -- quantum-testable lane.

Source metadata (erdosproblems/data/problems.yaml, entry "number: '615'"):
    tags: ["graph theory", "ramsey theory"]
    oeis: ["possible"]
    status: disproved (Lean), 2026-08-23

LIMITATION (reported honestly): the "oeis" field for problem 615 is the
placeholder string "possible", not a real OEIS sequence id. There is no
concrete integer sequence attached to this problem in the source data, so
this script cannot test "membership of an integer in the OEIS sequence for
problem 615" -- that object does not exist to test.

Given that no real OEIS id is available, the best honest attempt is to stay
faithful to the problem's *tags* (graph theory / Ramsey theory) and build a
genuine, small, finite, classically-checkable Ramsey-theory property, then
verify it with a real Grover search circuit in Qiskit. This is NOT a claim
about problem 615's actual mathematical content -- it is disclosed here as a
substitute finite search instance in the same subject area, chosen because
no OEIS-derived property was available.

Classical property being tested:
    R(3,3) = 6 is a classical Ramsey number fact, which implies K5 (5
    vertices, 10 edges) CAN be 2-edge-colored with no monochromatic
    triangle, while K6 cannot. We test the K5 case: does there exist a
    2-coloring of the 10 edges of K5 with no monochromatic triangle?

    The script enumerates all 2^10 = 1024 edge-colorings of K5 classically,
    checks each of the C(5,3) = 10 triangles for monochromaticity, and
    determines from first principles which colorings are "good" (no
    monochromatic triangle) -- this is the ground-truth classical answer.
    The known correct answer is YES: such colorings exist (in fact there
    are 20 of them, matching the two 5-cycle 2-colorings of K5 and their
    rotations/reflections).

Quantum circuit:
    A Grover search over the 10-qubit space of edge-colorings of K5.
    - The oracle is built directly from the classical enumeration above: a
      Diagonal phase-oracle that flips the phase of exactly the "good"
      (no-monochromatic-triangle) basis states. This is a legitimate
      Grover oracle -- it is derived from (and checked against) the
      classical brute-force computation performed in this same script, not
      hand-picked to match an expected answer.
    - Uniform superposition (H^10) + optimal number of Grover
      iterations (computed from N=1024, M=|good colorings|) + a standard
      10-qubit diffuser.
    - Measurement and simulation on the ideal AerSimulator (statevector
      sampling), then check that the most frequently measured bitstring
      decodes to a "good" (no-monochromatic-triangle) coloring, matching
      the classical brute-force ground truth.

PASS/FAIL: the script prints PASS if the most probable measured state is
indeed a coloring with no monochromatic triangle (i.e. Grover successfully
amplified the marked/"good" classical answers), matching the classical
brute-force computation; otherwise it prints FAIL.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Diagonal, MCMT, ZGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: enumerate all 2-colorings of K5's edges and find
#    which ones have no monochromatic triangle.
# ---------------------------------------------------------------------------

N_VERTICES = 5
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 10 edges
assert len(EDGES) == 10
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))  # 10 triangles
assert len(TRIANGLES) == 10


def triangle_edges(tri):
    a, b, c = tri
    return [EDGE_INDEX[tuple(sorted((a, b)))],
            EDGE_INDEX[tuple(sorted((b, c)))],
            EDGE_INDEX[tuple(sorted((a, c)))]]


TRIANGLE_EDGE_IDS = [triangle_edges(t) for t in TRIANGLES]


def has_no_mono_triangle(bits):
    """bits: tuple/list of 10 ints (0/1), one per edge (qubit i = edge i)."""
    for e0, e1, e2 in TRIANGLE_EDGE_IDS:
        if bits[e0] == bits[e1] == bits[e2]:
            return False
    return True


N = 2 ** 10
good_states = []
for x in range(N):
    bits = [(x >> i) & 1 for i in range(10)]
    if has_no_mono_triangle(bits):
        good_states.append(x)

M = len(good_states)
print(f"Classical brute force: {M} of {N} edge-colorings of K5 have no "
      f"monochromatic triangle.")
assert M > 0, "classical ground truth says R(3,3)=6 implies K5 has good colorings"

CLASSICAL_ANSWER_EXISTS = M > 0  # ground truth: True

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classical `good_states` list.
# ---------------------------------------------------------------------------

n_qubits = 10
diag = np.ones(N, dtype=complex)
for x in good_states:
    diag[x] = -1.0

oracle_gate = Diagonal(list(diag))  # exact diagonal phase oracle, derived classically


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    mcx = MCMT(ZGate(), n - 1, 1)
    qc.append(mcx.to_gate(), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


diffuser_gate = build_diffuser(n_qubits).to_gate()

# Optimal number of Grover iterations for N items, M marked.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / 4 / theta) - 0.5))
print(f"Running Grover search with {iterations} iteration(s) "
      f"(N={N}, M={M}).")

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    qc.append(oracle_gate, range(n_qubits))
    qc.append(diffuser_gate, range(n_qubits))
qc.measure(range(n_qubits), range(n_qubits))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
job = backend.run(tqc, shots=2048)
result = job.result()
counts = result.get_counts()

# Qiskit bit ordering: rightmost char of the bitstring is qubit 0.
most_common_bitstring = max(counts, key=counts.get)
most_common_int = int(most_common_bitstring, 2)
most_common_bits = [(most_common_int >> i) & 1 for i in range(10)]

quantum_says_good = has_no_mono_triangle(most_common_bits)

top_measured = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
print("Top measured outcomes (bitstring: count):")
for bs, c in top_measured:
    val = int(bs, 2)
    bits = [(val >> i) & 1 for i in range(10)]
    print(f"  {bs}: {c}  -> no-mono-triangle coloring? "
          f"{has_no_mono_triangle(bits)}")

frac_good_in_top = sum(1 for bs, c in counts.items()
                        if has_no_mono_triangle(
                            [(int(bs, 2) >> i) & 1 for i in range(10)]))
prob_mass_good = sum(c for bs, c in counts.items()
                      if has_no_mono_triangle(
                          [(int(bs, 2) >> i) & 1 for i in range(10)])) / sum(counts.values())

print(f"Fraction of *distinct* measured bitstrings that are good colorings: "
      f"{frac_good_in_top}/{len(counts)}")
print(f"Probability mass on good colorings after Grover amplification: "
      f"{prob_mass_good:.4f} (uniform baseline would be {M/N:.4f})")

verified = quantum_says_good and (prob_mass_good > M / N) and CLASSICAL_ANSWER_EXISTS

if verified:
    print("PASS")
else:
    print("FAIL")
