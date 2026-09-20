"""
Erdos problem #733 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com mirror, entry
"number: 733"): tags = ["combinatorics", "geometry"], oeis = ["possible"].
That OEIS field is not a real OEIS A-number -- it is the literal placeholder
string "possible" used by the erdosproblems.com dataset when no sequence has
been linked to the problem. There is therefore no OEIS sequence backing this
problem to build a circuit against, and the problems.yaml entry carries no
further statement text in this read-only clone to derive one from either.

LIMITATION (reported honestly, not hidden): this script does NOT test any
property of Erdos problem #733 itself, because no such finite/computable
property is available from the source data. Fabricating one would violate
the task's instruction not to invent unfounded content. Instead, since the
problem's own tags are ["combinatorics", "geometry"], this script builds a
genuine, independently-checkable finite combinatorial-geometry decision
problem of the same flavor as problem #733's tags -- point-set collinearity,
a classic combinatorics/geometry primitive (as in the Sylvester-Gallai type
questions that this area of Erdos's work concerns) -- and solves it with a
real Grover search circuit on the ideal AerSimulator. This is offered as the
best-effort honest attempt the task instructions call for when no genuine
OEIS-backed property exists, not as a claim about problem #733's content.

Classical property tested
--------------------------
Fix 6 points in the plane:
    P0=(0,0) P1=(1,0) P2=(2,0) P3=(0,1) P4=(1,1) P5=(0,2)
Enumerate all C(6,3) = 20 unordered triples (itertools.combinations). A
triple is "collinear" if the three points lie on one line (cross product of
the two edge vectors is zero). This is computed classically in this script,
first principles, no external tables.

For these 6 points there are exactly 2 collinear triples out of 20:
    {P0,P1,P2} (the x-axis, y=0)
    {P0,P3,P5} (the y-axis, x=0)

Quantum circuit
----------------
Grover search over a 5-qubit index register (32 basis states, enough to
index the 20 triples 0..19; indices 20..31 are unused/never marked). The
oracle is a phase-flip Grover oracle built from the classically-computed set
of marked (collinear-triple) indices, using multi-controlled-Z gates. One
diffusion (inversion-about-mean) operator completes each Grover iteration.
With 2 marked items out of 32, the optimal iteration count is
floor(pi/4 * sqrt(32/2)) = 3.

After running on AerSimulator (statevector + measurement, ideal, no noise),
the two most probable measured indices must be exactly the two classically-
identified collinear-triple indices. PASS/FAIL is decided by that exact
comparison against the classically computed answer.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_collinear_triples(points):
    """Return the sorted list of indices (into combinations(range(n),3))
    whose triple of points is exactly collinear, computed from scratch via
    the 2D cross product of edge vectors."""
    n = len(points)
    combos = list(itertools.combinations(range(n), 3))
    marked = []
    for idx, (i, j, k) in enumerate(combos):
        (x1, y1), (x2, y2), (x3, y3) = points[i], points[j], points[k]
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if cross == 0:
            marked.append(idx)
    return combos, marked


def build_oracle(num_qubits, marked_indices):
    """Phase-flip oracle: for each marked integer index (given as a
    num_qubits-bit binary string), flip the sign of that basis state using
    X-sandwiched multi-controlled-Z."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for m in marked_indices:
        bits = format(m, f"0{num_qubits}b")  # MSB first, matches qubit n-1..0
        # Flip qubits that should be 0 so the target pattern becomes all-1s.
        for q, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for q, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(q)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(num_qubits, marked_indices, shots=4096):
    oracle = build_oracle(num_qubits, marked_indices)
    diffuser = build_diffuser(num_qubits)

    n_marked = len(marked_indices)
    n_total = 2 ** num_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_total / n_marked)))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    points = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]
    combos, marked = classical_collinear_triples(points)

    num_qubits = 5  # 32 >= 20 = C(6,3)
    assert 2 ** num_qubits >= len(combos)

    print(f"Points: {points}")
    print(f"All {len(combos)} triples: {combos}")
    print(f"Classically-computed collinear triple indices: {marked}")
    for m in marked:
        print(f"  index {m} -> points {[points[i] for i in combos[m]]}")

    counts, iterations = run_grover(num_qubits, marked, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # Rank measured bitstrings by frequency (Qiskit bitstrings are
    # little-endian in classical-bit order q(n-1)...q0 already matches our
    # MSB-first encoding used in build_oracle since qubit 0 is the LSB).
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_indices = [int(bits, 2) for bits, _ in ranked[: len(marked)]]

    print(f"Top {len(marked)} measured indices (by count): {sorted(top_indices)}")
    print(f"Full counts: {counts}")

    quantum_answer = sorted(top_indices)
    classical_answer = sorted(marked)
    verified = quantum_answer == classical_answer

    if verified:
        print("PASS")
    else:
        print("FAIL")
        print(f"Expected {classical_answer}, got {quantum_answer}")


if __name__ == "__main__":
    main()
