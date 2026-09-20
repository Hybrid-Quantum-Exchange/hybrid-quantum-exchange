"""
Erdos problem #174 -- quantum-testable lane.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry
`number: "174"`, verified 2026-09-19):
    prize: no
    status: open (unformalized)
    oeis: ["N/A"]
    tags: ["geometry", "ramsey theory"]

LIMITATION (read before trusting the "PASS"):
Problem 174 has NO associated OEIS sequence ("N/A" in the source data), so
there is no OEIS-derived integer sequence to build a membership/term oracle
for, as the general instructions for this lane ask for. Per those
instructions ("if no OEIS id ... write the script anyway with your best
honest attempt, note the limitation clearly"), this script does not test
problem 174 itself. Instead it builds a genuine, finite, classically-checked
computation from the problem's own tag ("ramsey theory") that a real Grover
search can verify: a witness computation for the Ramsey number R(3,3) = 6.

The classical property under test:
    Take the 2-colouring of the complete graph K6 given by:
      - for i,j in {0,1,2,3,4} (i<j): colour(i,j) = RED  if (j-i) mod 5 in {1,4}
                                        colour(i,j) = BLUE otherwise
        (this is the standard pentagon/pentagram colouring of K5, which is
        exactly the extremal 2-colouring witnessing R(3,3) > 5, i.e. it has
        NO monochromatic triangle)
      - for the 6th vertex (index 5): colour(i,5) = RED if i is even else BLUE
    Ramsey's theorem guarantees R(3,3) = 6, so ANY 2-colouring of K6 must
    contain at least one monochromatic triangle. The classical property
    tested here is: "does this specific, fully-defined 2-colouring of K6
    contain a monochromatic triangle, and if so, which one(s)?"

    This is computed directly in Python (first principles, brute force over
    all C(6,3) = 20 triangles) to get the classical ground truth: the exact
    set of monochromatic-triangle indices.

The quantum circuit:
    Grover's search algorithm over the 20 triangles of K6 (indices 0..19,
    encoded in 5 qubits, 32 basis states). The oracle is built directly from
    the classical monochromatic-triangle set computed above (a diagonal
    phase oracle marking exactly those basis states), and amplitude
    amplification is run for the optimal number of Grover iterations. The
    circuit is executed on the ideal AerSimulator and the most frequently
    measured index is compared against the classical set of monochromatic
    triangles.

PASS criterion: the index most frequently measured by the Grover circuit is
one of the classically-computed monochromatic triangles of this K6
colouring (i.e. quantum search found a real, verified witness for R(3,3)=6
on this instance).
"""

import itertools
import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: build the K6 2-colouring and find all
#    monochromatic triangles by brute-force enumeration.
# ---------------------------------------------------------------------------

def colour(i, j):
    """RED/BLUE colour of edge (i,j) in the fixed K6 2-colouring."""
    a, b = min(i, j), max(i, j)
    if b < 5:
        # pentagon/pentagram colouring on vertices {0,1,2,3,4}
        return "RED" if (b - a) % 5 in (1, 4) else "BLUE"
    else:
        # edges touching the 6th vertex (index 5)
        return "RED" if a % 2 == 0 else "BLUE"


VERTICES = list(range(6))
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 20 triangles, index 0..19
assert len(TRIANGLES) == 20

def is_monochromatic(tri):
    a, b, c = tri
    cols = {colour(a, b), colour(b, c), colour(a, c)}
    return len(cols) == 1

MONOCHROMATIC_INDICES = sorted(
    idx for idx, tri in enumerate(TRIANGLES) if is_monochromatic(tri)
)

print("Classical brute-force result:")
print(f"  {len(TRIANGLES)} triangles in K6, checked directly.")
print(f"  Monochromatic triangle indices: {MONOCHROMATIC_INDICES}")
for idx in MONOCHROMATIC_INDICES:
    tri = TRIANGLES[idx]
    print(f"    index {idx}: vertices {tri}, colour {colour(tri[0], tri[1])}")

# Ramsey's theorem (R(3,3) = 6) guarantees this set is non-empty for ANY
# 2-colouring of K6; verify that classically too, as a sanity check on the
# construction itself (not on the quantum part).
assert len(MONOCHROMATIC_INDICES) >= 1, (
    "Construction error: R(3,3)=6 guarantees a monochromatic triangle in "
    "every 2-colouring of K6, but none was found -- the colouring above "
    "must be miscoded."
)

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 20 triangle indices (5 qubits,
#    32 basis states) for a monochromatic-triangle index.
# ---------------------------------------------------------------------------

N_QUBITS = 5
N_STATES = 2 ** N_QUBITS  # 32
MARKED = MONOCHROMATIC_INDICES
M = len(MARKED)


def apply_oracle(qc, qubits):
    """Phase-flip every basis state whose integer value is in MARKED."""
    for idx in MARKED:
        bits = format(idx, f"0{N_QUBITS}b")
        # Flip qubits that should be 0 so the marked pattern becomes all-1s.
        for q, bit in zip(qubits, bits):
            if bit == "0":
                qc.x(q)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q, bit in zip(qubits, bits):
            if bit == "0":
                qc.x(q)


def apply_diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))

# Uniform superposition over all 32 basis states.
qc.h(qubits)

# Optimal number of Grover iterations for N=32, M=len(MARKED).
n_iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M) - 0.5))

for _ in range(n_iterations):
    apply_oracle(qc, qubits)
    apply_diffuser(qc, qubits)

qc.measure(qubits, qubits)

print(f"\nGrover circuit: {N_QUBITS} qubits, {N_STATES} states, "
      f"{M} marked item(s), {n_iterations} Grover iteration(s).")

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order is little-endian in the classical register string
# (qc.measure(qubits, qubits) maps qubit i -> classical bit i, and the
# returned bitstring lists classical bits high-to-low), so reverse before
# converting to an integer index.
index_counts = Counter()
for bitstring, count in counts.items():
    idx = int(bitstring[::-1], 2)
    index_counts[idx] += count

most_common_idx, most_common_count = index_counts.most_common(1)[0]

print(f"\nQuantum measurement: most frequent index = {most_common_idx} "
      f"({most_common_count}/{shots} shots)")
print(f"Classical monochromatic-triangle indices: {MARKED}")

verified = most_common_idx in MARKED

if verified:
    print("PASS")
else:
    print("FAIL")

assert verified, (
    "Grover search did not converge on a classically-verified "
    "monochromatic triangle of the K6 colouring."
)
