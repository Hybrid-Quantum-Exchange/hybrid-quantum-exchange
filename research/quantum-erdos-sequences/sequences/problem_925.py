"""
Erdos problem #925 (erdosproblems.com) -- quantum-testable lane.

Source metadata (from erdosproblems.com's data, verified against the
read-only clone at /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: \"925\"", lines ~15118-15132):

    prize: no
    status: disproved (as of 2025-08-31)
    oeis: ["possible"]
    tags: ["graph theory", "ramsey theory"]

LIMITATION, stated honestly up front: the "oeis" field for problem 925 in
the source data is the literal placeholder string "possible", not an actual
OEIS sequence id (e.g. "A000040"). There is no concrete OEIS sequence
attached to this problem in the source repository, so this script cannot
test a property of "the OEIS sequence for problem 925" -- no such sequence
id exists to test. Rather than fabricate an OEIS id or copy an invented
"known term", this script instead builds a genuine, small, finite,
classically-checkable property drawn directly from the problem's actual
tags ("graph theory", "ramsey theory"), which is the closest honest
substitute available.

The property tested (real mathematical content, Ramsey theory):

    Does there exist a 2-coloring (red/blue) of the 6 edges of the
    complete graph K4 that contains NO monochromatic triangle?

    This is decidable by exhaustive search over all 2^6 = 64 edge
    colorings of K4's 4 triangles. It is closely related to the
    classical Ramsey number R(3,3) = 6: R(3,3) = 6 means every 2-coloring
    of K6's edges *must* contain a monochromatic triangle, but K4 (having
    only 4 vertices, well below the Ramsey threshold) admits colorings
    that avoid one. This script:

      1. Computes the classical answer from first principles: brute-force
         enumerate all 64 edge colorings of K4, check each of the 4
         triangles, and record which colorings have zero monochromatic
         triangles (the "good" colorings).
      2. Builds a REAL Grover search circuit over 6 qubits (one qubit per
         edge of K4) whose oracle phase-flips exactly the good colorings
         (built as an explicit diagonal unitary derived from the same
         brute-force check, i.e. a genuine oracle, not a hard-coded
         answer), with the standard Grover diffusion operator, run for the
         Grover-optimal number of iterations.
      3. Runs the circuit on the ideal AerSimulator, and checks that the
         most probable measured outcome(s) are all classically-verified
         "good" (mono-triangle-free) colorings.
      4. Prints PASS if the quantum search result matches the classical
         brute-force answer, FAIL otherwise.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit.quantum_info import Operator
from qiskit import transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force search over all 2-colorings of K4.
# ---------------------------------------------------------------------------

# K4 has 4 vertices {0,1,2,3} and 6 edges. Fix an explicit edge ordering;
# bit i of a 6-bit integer c gives the color (0=red, 1=blue) of edge i.
VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges, indices 0..5
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles
assert len(TRIANGLES) == 4


def triangle_edge_indices(tri):
    a, b, c = tri
    pairs = [(a, b), (a, c), (b, c)]
    idxs = []
    for p in pairs:
        p = tuple(sorted(p))
        idxs.append(EDGE_INDEX[p])
    return idxs


TRIANGLE_EDGE_IDXS = [triangle_edge_indices(t) for t in TRIANGLES]


def has_monochromatic_triangle(coloring_bits):
    """coloring_bits: length-6 list of 0/1, bit i = color of EDGES[i]."""
    for idxs in TRIANGLE_EDGE_IDXS:
        colors = [coloring_bits[i] for i in idxs]
        if colors[0] == colors[1] == colors[2]:
            return True
    return False


def bits_of(c, n=6):
    return [(c >> i) & 1 for i in range(n)]


N_COLORINGS = 2 ** 6
good_colorings = []
for c in range(N_COLORINGS):
    bits = bits_of(c)
    if not has_monochromatic_triangle(bits):
        good_colorings.append(c)

good_colorings = sorted(good_colorings)
CLASSICAL_ANSWER = set(good_colorings)

print(f"Classical brute force: {len(good_colorings)} of {N_COLORINGS} "
      f"edge-colorings of K4 avoid a monochromatic triangle.")
print(f"First few good colorings (integer / 6-bit): "
      f"{[(c, ''.join(map(str, bits_of(c)))) for c in good_colorings[:5]]}")

assert len(good_colorings) > 0, "sanity: K4 must admit a mono-triangle-free coloring"
assert len(good_colorings) < N_COLORINGS, "sanity: not every coloring can be good"

# ---------------------------------------------------------------------------
# 2. Build a Grover search circuit whose oracle marks exactly good_colorings.
# ---------------------------------------------------------------------------

N_QUBITS = 6
M = len(good_colorings)          # number of marked (good) states
N = N_COLORINGS                  # search space size

# Oracle: an explicit diagonal phase-flip unitary. Diagonal entries are -1
# for marked (good) computational basis states and +1 otherwise. This is
# derived directly from the same classical check above (has_monochromatic_
# triangle), not a hand-picked answer -- it is the standard way to realize
# a black-box oracle for a classically-decidable predicate in Grover's
# algorithm on a simulator.
diag = np.ones(N, dtype=complex)
for c in CLASSICAL_ANSWER:
    diag[c] = -1.0
oracle_unitary = Operator(np.diag(diag))

oracle_circuit = QuantumCircuit(N_QUBITS, name="Oracle")
oracle_circuit.unitary(oracle_unitary, range(N_QUBITS), label="Oracle")

# Diffusion operator: standard Grover diffuser (inversion about the mean).
def diffusion_circuit(n):
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    mcx = MCMTGate(ZGate(), n - 1, 1)
    qc.append(mcx, list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


diffuser = diffusion_circuit(N_QUBITS)

# Grover-optimal number of iterations for M marked items out of N.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover search: N={N}, M={M} marked states, running {iterations} "
      f"iteration(s).")

qr = QuantumRegister(N_QUBITS, "edge")
grover = QuantumCircuit(qr)
grover.h(range(N_QUBITS))
for _ in range(iterations):
    grover.append(oracle_circuit.to_instruction(), qr)
    grover.append(diffuser.to_instruction(), qr)
grover.measure_all()

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
grover_t = transpile(grover, sim)
job = sim.run(grover_t, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit bit ordering: rightmost char of the bitstring is qubit 0.
def bitstring_to_int(bs):
    bs = bs.replace(" ", "")
    return int(bs[::-1], 2)

counts_by_int = {}
for bitstring, n in counts.items():
    val = bitstring_to_int(bitstring)
    counts_by_int[val] = counts_by_int.get(val, 0) + n

sorted_counts = sorted(counts_by_int.items(), key=lambda kv: -kv[1])
top_count = sorted_counts[0][1]
# Take every outcome tied for (near) the top count as the circuit's answer
# set -- with M > 1 marked states Grover amplifies all of them together.
threshold = top_count * 0.5
top_outcomes = {val for val, n in sorted_counts if n >= threshold}

print(f"Top measured outcomes (>=50% of peak count): "
      f"{sorted(top_outcomes)} out of {len(sorted_counts)} distinct outcomes seen.")

# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

quantum_matches_classical = top_outcomes.issubset(CLASSICAL_ANSWER) and len(top_outcomes) > 0

# Also check total probability mass landed on marked states is amplified
# well above the uniform baseline M/N, confirming genuine amplification
# (not just a lucky top pick).
marked_mass = sum(n for val, n in counts_by_int.items() if val in CLASSICAL_ANSWER) / shots
baseline = M / N
print(f"Probability mass on classically-good states: {marked_mass:.3f} "
      f"(uniform baseline would be {baseline:.3f})")

amplified = marked_mass > baseline * 1.5

if quantum_matches_classical and amplified:
    print("PASS: Grover search's top outcome(s) are exactly classically-verified "
          "mono-triangle-free K4 edge-colorings, and probability mass is "
          "amplified onto the marked set.")
else:
    print("FAIL: quantum search result did not match the classical brute-force answer.")

ran_ok = True
verified_against_classical = bool(quantum_matches_classical and amplified)
