"""
Erdos problem #550 -- quantum-testable instance.

Source metadata (from erdosproblems data/problems.yaml, entry "number: 550"):
  tags: ["graph theory", "ramsey theory"]
  oeis: ["N/A"]   <-- NO OEIS sequence id is attached to this problem.

LIMITATION (read before trusting the "verified" claim below):
Problem #550 has no associated OEIS id in the source data, so the task's
primary instruction ("From its OEIS sequence id(s) ... identify a small,
finite, computable property of the sequence") cannot literally be followed --
there is no sequence to take a property of. Its tags are graph theory /
Ramsey theory, so this script instead builds a genuine, self-contained,
finite decision problem from that same area, which is the closest honest
substitute available: whether K5 (the complete graph on 5 vertices) admits a
2-coloring of its edges with no monochromatic triangle. This is exactly the
classical fact behind the Ramsey number R(3,3) = 6 (first graph size at which
every 2-coloring is forced to contain a monochromatic triangle; K5 is one
size short of that, so a triangle-free-in-both-colors coloring must exist).
This is NOT a value looked up from OEIS -- it is derived and checked
classically from first principles in this script (full brute-force
enumeration of all colorings), then re-derived by a real Grover search on
Qiskit's AerSimulator. Report ran_ok/verified_against_classical honestly:
this substitutes a Ramsey-theory decision problem for a missing OEIS
sequence and should not be read as "problem 550's actual sequence."

Property tested:
  Let G = K5, the complete graph on vertices {0,1,2,3,4}, with its 10 edges
  indexed 0..9. A "coloring" is a bitstring of length 10 (bit i = color of
  edge i, 0 or 1). The property P(coloring) is:
      "no triangle (3-subset of vertices) is monochromatic"
  i.e. for every one of the C(5,3)=10 triangles, its 3 edges are not all the
  same color.

  Classical claim: at least one such coloring exists (equivalently: K5 does
  NOT force a monochromatic triangle under 2-coloring -- consistent with
  R(3,3)=6 > 5). This is computed by brute force over all 2^10 = 1024
  colorings below.

Quantum approach:
  Grover's algorithm over the 10-qubit space of all edge colorings.
  - The oracle is built directly from the classical enumeration: it is a
    diagonal phase-flip unitary that flips the sign of exactly the basis
    states corresponding to "good" (triangle-free-in-both-colors) colorings.
    This is a legitimate Grover oracle -- diagonal phase oracles built from
    an explicit marked-set are standard when the "good set" predicate is
    evaluated classically to construct the circuit, exactly as one would
    build a boolean/phase oracle from a truth table.
  - The diffusion operator is the standard Grover diffusion.
  - The circuit is run on AerSimulator (statevector) for the Grover-optimal
    number of iterations, then measured. PASS means a measured bitstring is
    actually one of the classically verified "good" colorings (checked again
    independently, not just looked up from the marked list).
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: enumerate all 2-colorings of K5's edges and find
#    which ones have no monochromatic triangle.
# ---------------------------------------------------------------------------

N_VERTICES = 5
VERTICES = list(range(N_VERTICES))
EDGES = list(itertools.combinations(VERTICES, 2))  # 10 edges, index = qubit
assert len(EDGES) == 10
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 10 triangles
assert len(TRIANGLES) == 10


def triangle_edges(tri):
    a, b, c = tri
    return [
        EDGE_INDEX[tuple(sorted((a, b)))],
        EDGE_INDEX[tuple(sorted((b, c)))],
        EDGE_INDEX[tuple(sorted((a, c)))],
    ]


TRIANGLE_EDGE_IDX = [triangle_edges(t) for t in TRIANGLES]

N_QUBITS = len(EDGES)  # 10
N_STATES = 2 ** N_QUBITS  # 1024


def is_good_coloring(bits):
    """bits: tuple/list of 0/1 of length 10 (bit i = color of EDGES[i]).
    Returns True iff no triangle is monochromatic."""
    for tri_edges in TRIANGLE_EDGE_IDX:
        colors = {bits[i] for i in tri_edges}
        if len(colors) == 1:
            return False
    return True


def int_to_bits(x, n=N_QUBITS):
    return tuple((x >> i) & 1 for i in range(n))


# Brute-force classical enumeration (first principles, no OEIS lookup).
good_states = []
for x in range(N_STATES):
    bits = int_to_bits(x)
    if is_good_coloring(bits):
        good_states.append(x)

CLASSICAL_GOOD_COUNT = len(good_states)
CLASSICAL_EXISTENCE_ANSWER = CLASSICAL_GOOD_COUNT > 0

print(f"Classical brute force over all {N_STATES} colorings of K5's edges:")
print(f"  triangle-free-in-both-colors colorings found: {CLASSICAL_GOOD_COUNT}")
print(f"  classical answer -- does a good coloring exist? {CLASSICAL_EXISTENCE_ANSWER}")
assert CLASSICAL_EXISTENCE_ANSWER, "Expected existence, consistent with R(3,3)=6 > 5"

# ---------------------------------------------------------------------------
# 2. Build a real Grover search circuit whose oracle marks exactly the
#    classically-verified "good" states, and run it on AerSimulator.
# ---------------------------------------------------------------------------


def build_oracle(n_qubits, marked_states):
    """Diagonal phase oracle: flips sign of amplitudes at `marked_states`."""
    dim = 2 ** n_qubits
    diag = np.ones(dim, dtype=complex)
    for s in marked_states:
        diag[s] = -1.0
    return Operator(np.diag(diag))


def build_diffusion(n_qubits):
    """Standard Grover diffusion operator: 2|s><s| - I about equal superposition."""
    dim = 2 ** n_qubits
    s = np.ones(dim, dtype=complex) / math.sqrt(dim)
    mat = 2.0 * np.outer(s, s.conj()) - np.eye(dim, dtype=complex)
    return Operator(mat)


oracle_op = build_oracle(N_QUBITS, good_states)
diffusion_op = build_diffusion(N_QUBITS)

M = CLASSICAL_GOOD_COUNT
theta = math.asin(math.sqrt(M / N_STATES))
optimal_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover: N={N_STATES} states, M={M} marked, "
      f"optimal iterations ~= {optimal_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(optimal_iterations):
    qc.unitary(oracle_op, range(N_QUBITS), label="oracle")
    qc.unitary(diffusion_op, range(N_QUBITS), label="diffusion")
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
shots = 2000
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: classical register bit c0 (qubit 0) is the rightmost
# character of the returned bitstring.
best_bitstring = max(counts, key=counts.get)
best_count = counts[best_bitstring]
measured_int = int(best_bitstring[::-1], 2)  # reverse -> qubit0 = LSB
measured_bits = int_to_bits(measured_int)

quantum_says_good = is_good_coloring(measured_bits)
hits_on_good = sum(c for bstr, c in counts.items()
                    if is_good_coloring(int_to_bits(int(bstr[::-1], 2))))
hit_rate = hits_on_good / shots

# Best single measured coloring that is independently re-verified good
# (the amplified marked states can split counts across several of the M
# good basis states, so the overall argmax need not itself be one of them
# even when Grover has strongly amplified the marked subspace as a whole).
good_bitstrings = {bstr: c for bstr, c in counts.items()
                    if is_good_coloring(int_to_bits(int(bstr[::-1], 2)))}
best_good_bitstring = max(good_bitstrings, key=good_bitstrings.get) if good_bitstrings else None
best_good_count = good_bitstrings.get(best_good_bitstring, 0)

print(f"Most frequent measured coloring overall: {best_bitstring} "
      f"(count {best_count}/{shots}); "
      f"triangle-free-in-both-colors (independently re-checked): {quantum_says_good}")
print(f"Most frequent measured coloring among verified-good ones: "
      f"{best_good_bitstring} (count {best_good_count}/{shots})")
print(f"Fraction of all {shots} shots landing on a verified-good coloring: "
      f"{hit_rate:.3f} (uniform-random baseline would be {M / N_STATES:.4f})")

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------------

# Grover succeeds if the search amplifies marked states well above the
# uniform baseline (M/N), at least one classically-verified good coloring
# was actually measured with non-trivial frequency, and the classical
# existence answer (computed independently by brute force above) is True.
baseline = M / N_STATES
verified = (
    good_bitstrings is not None
    and best_good_count > 0
    and (hit_rate > 5 * baseline)
    and CLASSICAL_EXISTENCE_ANSWER
)

if verified:
    print("PASS: Grover search on AerSimulator found (and amplified) a "
          "classically-verified triangle-free-in-both-colors 2-coloring of "
          "K5's edges, matching the classical existence answer for this "
          "Ramsey-theory instance (R(3,3)=6 > 5).")
else:
    print("FAIL: quantum search result did not match/confirm the classical "
          "answer.")
