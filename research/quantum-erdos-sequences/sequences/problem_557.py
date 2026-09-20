"""
Erdos problem #557 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com mirror at
manman4/erdosproblems): problem 557 is tagged ["graph theory", "ramsey
theory"], status "proved (Lean)", and its `oeis` field is literally
`["N/A"]` -- there is NO OEIS sequence attached to this problem in the
source data. That means the "identify a property from its OEIS id(s)"
instruction cannot be followed literally: there is no id to derive a
property from.

LIMITATION (stated honestly, per instructions): this script does not test
any OEIS sequence for problem 557, because none exists in the source
metadata. Instead, since the problem's tags are graph theory / Ramsey
theory, this script builds a genuine, self-contained, classically-checked
Ramsey-theory instance in the same spirit as the problem's subject matter,
and tests it with a real Grover search circuit. This is offered as the
best honest attempt available given the missing OEIS id, not as a claim
that it tests problem 557's actual content.

Classical property tested
--------------------------
Ramsey number R(3,3) = 6 means: K_5 (5 vertices, 10 edges) CAN be
2-colored (red/blue) with no monochromatic triangle, but K_6 cannot.
This script:

  1. Enumerates all 2^10 = 1024 possible 2-colorings of the 10 edges of
     K_5.
  2. For each coloring, classically checks all C(5,3) = 10 triangles and
     flags the coloring as "good" iff no triangle is monochromatic.
  3. This is computed from first principles in `classical_good_colorings`
     below -- no literal OEIS/known value is copied in.
  4. Builds a 10-qubit Grover search circuit whose oracle marks exactly
     the "good" colorings (the oracle's marked set is derived from the
     classical enumeration in step 1-2, embedded as a diagonal phase
     oracle -- a standard technique for small, classically-enumerable
     black-box functions), and searches for one.
  5. Runs the circuit on AerSimulator, measures, and checks (classically)
     that the measured 10-bit coloring is indeed monochromatic-triangle
     free -- i.e. that Grover search found a true member of the good set.
     Compares against the classically known truth "a good coloring of K_5
     exists" (True, since R(3,3) = 6 > 5).

PASS/FAIL: prints PASS iff the quantum-measured coloring is confirmed
(classically) to be a valid triangle-free 2-coloring of K_5, matching the
classical fact that such colorings exist.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, MCXGate
from qiskit import transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical setup: edges and triangles of K_5
# ---------------------------------------------------------------------

VERTICES = range(5)
EDGES = list(itertools.combinations(VERTICES, 2))  # 10 edges
assert len(EDGES) == 10
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 10 triangles
assert len(TRIANGLES) == 10


def triangle_edges(tri):
    a, b, c = tri
    return (EDGE_INDEX[(a, b)], EDGE_INDEX[(a, c)], EDGE_INDEX[(b, c)])


TRIANGLE_EDGE_IDX = [triangle_edges(t) for t in TRIANGLES]


def is_good_coloring(bits):
    """bits: tuple of 10 ints (0=red,1=blue), one per edge in EDGES order.
    Returns True iff no triangle is monochromatic."""
    for e1, e2, e3 in TRIANGLE_EDGE_IDX:
        if bits[e1] == bits[e2] == bits[e3]:
            return False
    return True


def classical_good_colorings():
    """Brute-force, from first principles, every 2-coloring of K_5's edges
    and return the set of integer indices (0..1023) of colorings with no
    monochromatic triangle."""
    good = []
    for n in range(1024):
        bits = tuple((n >> i) & 1 for i in range(10))
        if is_good_coloring(bits):
            good.append(n)
    return good


GOOD = classical_good_colorings()
CLASSICAL_ANSWER_EXISTS = len(GOOD) > 0  # True since R(3,3) = 6 > 5
print(f"Classical brute force: {len(GOOD)} / 1024 colorings of K_5 are "
      f"monochromatic-triangle-free (expect > 0, since R(3,3)=6).")
assert CLASSICAL_ANSWER_EXISTS, "classical enumeration disagrees with R(3,3)=6"


# ---------------------------------------------------------------------
# 2. Quantum: Grover search for a "good" coloring
# ---------------------------------------------------------------------

N_QUBITS = 10
N_STATES = 2 ** N_QUBITS
M = len(GOOD)  # number of marked ("good") states


def build_oracle():
    """Diagonal phase oracle: -1 on every index in GOOD, +1 elsewhere.
    The marked set GOOD was derived purely classically above (step 1);
    this just encodes that same classical truth table as phases."""
    diag = np.ones(N_STATES, dtype=complex)
    for idx in GOOD:
        diag[idx] = -1.0
    return DiagonalGate(diag.tolist())


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover_circuit(iterations):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    oracle = build_oracle()
    diffuser = build_diffuser(N_QUBITS)
    for _ in range(iterations):
        qc.append(oracle, range(N_QUBITS))
        qc.append(diffuser.to_instruction(), range(N_QUBITS))
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


# Optimal Grover iteration count for N states, M marked
optimal_iterations = max(1, round((np.pi / 4) * np.sqrt(N_STATES / M)))
print(f"Marked states M={M}, running Grover with "
      f"{optimal_iterations} iteration(s) over {N_QUBITS} qubits.")

qc = grover_circuit(optimal_iterations)

sim = AerSimulator()
qc = transpile(qc, sim, basis_gates=None)
job = sim.run(qc, shots=256, seed_simulator=42)
result = job.result()
counts = result.get_counts()

# Most frequent measured outcome
best_bitstring = max(counts, key=counts.get)
# Qiskit bit order: rightmost char = qubit 0
measured_bits = tuple(int(best_bitstring[::-1][i]) for i in range(N_QUBITS))
measured_index = int(best_bitstring, 2)

hit_rate = sum(c for k, c in counts.items() if int(k, 2) in GOOD) / sum(counts.values())
print(f"Most frequent measured outcome: index {measured_index} "
      f"(count {counts[best_bitstring]}/256); "
      f"fraction of shots landing on a marked ('good') state: {hit_rate:.3f}")


# ---------------------------------------------------------------------
# 3. Verify quantum result against the classical answer
# ---------------------------------------------------------------------

quantum_found_good = is_good_coloring(measured_bits)
verified = quantum_found_good and CLASSICAL_ANSWER_EXISTS and hit_rate > 0.5

print(f"Quantum-measured coloring index {measured_index} is "
      f"{'a valid' if quantum_found_good else 'NOT a valid'} "
      f"monochromatic-triangle-free 2-coloring of K_5 "
      f"(cross-checked classically against `is_good_coloring`).")

if verified:
    print("PASS")
    sys.exit(0)
else:
    print("FAIL")
    sys.exit(1)
