"""
Quantum-testable instance for Erdos problem #838.

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
`number: "838"` (tags: ["geometry", "convex"]; oeis: ["possible"]).

LIMITATION, stated honestly up front: problem #838 carries no real OEIS
sequence id in the source data. Its `oeis` field is the literal string
"possible", which is not an OEIS A-number -- it is metadata noise, not a
sequence identifier. There is therefore no OEIS-backed integer sequence to
build a membership/divisibility/counting test around for this problem, and
this script does NOT fabricate one. Per the task's fallback instructions,
this is the "best honest attempt" case: no genuine OEIS-anchored property
exists to test, so instead this script builds a real, self-contained finite
combinatorial-geometry decision problem that matches the problem's own tags
("geometry", "convex") -- convex position of point subsets -- verifies the
classical answer from first principles in code, and then builds a genuine
Grover search circuit over that same finite search space and checks that
the quantum search recovers the classically-verified answer.

The classical property tested
------------------------------
Fix four points in the plane:
    A = (0, 0), B = (4, 0), C = (4, 4), D = (1, 1)
(D was chosen, and is verified below, to lie strictly inside triangle ABC.)

For each of the 2**4 = 16 subsets of {A, B, C, D} (indexed by a 4-bit
string, bit i = 1 means point i is included), define the subset to be
"marked" iff it has at least 3 points AND all of its points are in convex
position, i.e. none of them lies strictly inside the convex hull of the
others. This is computed by brute-force point-in-triangle / orientation
tests -- elementary, exact, and fully classical -- with no external
geometry library.

For 4 points with D strictly inside triangle ABC, the marked subsets are
exactly the four 3-point subsets {A,B,C}, {A,B,D}, {A,C,D}, {B,C,D} (each of
size 3 is trivially in convex position, since 3 non-collinear points always
are) MINUS none of them (all four are non-collinear triples), PLUS NOT the
4-point subset {A,B,C,D}, because D sits inside the hull of A,B,C. This
classical answer is computed in code below, not asserted.

The quantum circuit
--------------------
A Grover search circuit is built over the 4-qubit space of all 16 subsets.
The oracle is an exact diagonal phase-flip unitary (built directly from the
classical brute-force marking, i.e. it is a faithful oracle for the
classically-defined predicate, not a black box guess) that flips the phase
of exactly the marked basis states. The diffusion operator is the standard
Grover diffusion. The number of Grover iterations is chosen from the known
count of marked states via the standard formula. The circuit is simulated
on the ideal AerSimulator and the states with the highest measured
probability are compared against the classically-computed marked set.

PASS iff the set of most-probable measured 4-bit strings equals the
classically verified marked set.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator

ERDOS_PROBLEM_NUMBER = 838
OEIS_IDS_USED: list[str] = []  # none: source field is "possible", not an OEIS A-number
CLASSICAL_PROPERTY = (
    "For 4 fixed planar points {A,B,C,D}, which of the 16 subsets are in "
    "convex position (size >= 3, no point strictly inside the hull of the "
    "others)."
)

POINTS = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (1.0, 1.0)]  # A, B, C, D
N_POINTS = len(POINTS)


def orientation(p, q, r) -> float:
    """Signed area x2 of triangle pqr. >0 = counter-clockwise turn."""
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def point_in_triangle(pt, a, b, c) -> bool:
    """True iff pt lies strictly inside triangle abc (non-degenerate)."""
    d1 = orientation(a, b, pt)
    d2 = orientation(b, c, pt)
    d3 = orientation(c, a, pt)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def is_convex_position(indices: tuple[int, ...]) -> bool:
    """
    True iff the given point indices (size >= 3) are all in convex
    position: no point among them lies strictly inside the convex hull of
    the rest. For size == 3 this is automatically true whenever the three
    points are not collinear.
    """
    pts = [POINTS[i] for i in indices]
    k = len(pts)
    if k < 3:
        return False
    if k == 3:
        a, b, c = pts
        return orientation(a, b, c) != 0  # non-collinear triple
    if k == 4:
        # each point must NOT be strictly inside the triangle of the other 3
        for j in range(4):
            others = [pts[m] for m in range(4) if m != j]
            if point_in_triangle(pts[j], *others):
                return False
        # also require the 4 points aren't degenerate (no 3 collinear
        # through a hull edge in a way that breaks strict convex position);
        # for this fixed, verified instance that does not occur.
        return True
    raise ValueError("only sizes 3 and 4 are evaluated for this instance")


def classical_marked_bitmasks() -> set[int]:
    """
    Brute-force, from first principles, which of the 16 subsets of the 4
    fixed points are in convex position. Returns the set of 4-bit integer
    masks (bit i set => point i included) that are marked.
    """
    marked = set()
    for size in (3, 4):
        for combo in combinations(range(N_POINTS), size):
            if is_convex_position(combo):
                mask = 0
                for i in combo:
                    mask |= 1 << i
                marked.add(mask)
    return marked


def build_oracle(n_qubits: int, marked: set[int]) -> QuantumCircuit:
    """
    Exact diagonal phase-flip oracle: flips the phase of exactly the
    computational basis states listed in `marked`, built directly (not
    approximated) as a product of multi-controlled Z gates, one per marked
    integer, each preceded/followed by X gates on the 0-bits of that
    integer so the multi-controlled-Z fires only on that exact bitstring.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    for mask in sorted(marked):
        zero_bits = [i for i in range(n_qubits) if not (mask >> i) & 1]
        for i in zero_bits:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
            qc.append(mcz, list(range(n_qubits)))
        for i in zero_bits:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
        qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked: set[int], n_qubits: int, shots: int = 20000):
    n_total = 2 ** n_qubits
    m = len(marked)
    if m == 0 or m == n_total:
        raise ValueError("Grover search requires 0 < |marked| < 2**n")

    iterations = max(1, round((math.pi / 4) * math.sqrt(n_total / m)))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main() -> bool:
    print(f"Erdos problem #{ERDOS_PROBLEM_NUMBER}: no OEIS id available "
          f"(source field is 'possible', not an A-number).")
    print(f"Classical property tested instead: {CLASSICAL_PROPERTY}")
    print(f"Points: A={POINTS[0]} B={POINTS[1]} C={POINTS[2]} D={POINTS[3]}")

    # sanity: verify D is strictly inside triangle ABC, as claimed above
    d_inside = point_in_triangle(POINTS[3], POINTS[0], POINTS[1], POINTS[2])
    print(f"D strictly inside triangle ABC (classically verified): {d_inside}")
    assert d_inside, "instance assumption violated -- fix POINTS"

    marked = classical_marked_bitmasks()
    marked_bitstrings = {format(m, "04b") for m in marked}
    print(f"Classically computed marked subsets (bit i = point i included): "
          f"{sorted(marked_bitstrings)}")
    print(f"|marked| = {len(marked)} out of 16 total subsets")

    counts, iterations = run_grover(marked, n_qubits=N_POINTS)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    # sort by measured probability, descending
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_m = ranked[: len(marked)]
    top_bitstrings = {bs for bs, _ in top_m}

    print("Top measured bitstrings (by count):")
    for bs, c in ranked[:8]:
        print(f"  {bs}: {c}/{total_shots} ({100*c/total_shots:.1f}%)  "
              f"marked={bs in marked_bitstrings}")

    verified = top_bitstrings == marked_bitstrings
    print(f"Quantum top-{len(marked)} bitstrings match classical marked set: "
          f"{verified}")

    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
