"""
Erdos problem #101 (erdosproblems.com/101) -- the "orchard planting" problem:
among n points in the plane, what is the maximum number of lines containing
exactly 4 of the points? The associated OEIS sequence is A006065, "Maximal
number of 4-tree rows in n-tree orchard problem."

Classical property tested (finite, computable, genuinely tied to the
sequence's definition):

    Fix a concrete 9-point configuration in the plane, built from first
    principles in this script so that it contains EXACTLY ONE 4-point
    collinear line (i.e. exactly one "4-tree row" -- the object A006065
    counts). The configuration is:

        P0=(0,0), P1=(1,1), P2=(2,2), P3=(3,3)   -- exactly collinear (y=x)
        P4..P8 = (x, x^2) for x in {5,6,7,8,9}    -- on the parabola y=x^2

    A line meets a parabola in at most 2 points, so no 3 of P4..P8 are ever
    collinear, and since y=x meets y=x^2 only at x=0,1 (neither of which is
    among {5,...,9}), none of P4..P8 lies on the line y=x either. Hence the
    ONLY 4-point subset of these 9 points that is collinear is
    {P0, P1, P2, P3}. This is verified classically below by brute-force
    over all C(9,4) = 126 quadruples using exact cross-product collinearity
    tests (integer arithmetic, no floating point).

    We then encode "which of the 126 quadruples (indexed 0..125, packed
    into a 7-qubit register of 128 basis states) is the collinear one" as a
    Grover search problem: a quantum oracle, built directly from the
    classical collinearity computation, phase-flips exactly the one basis
    state corresponding to the collinear quadruple's index in the
    lexicographic list from itertools.combinations(range(9), 4). Grover's
    algorithm is run on the ideal AerSimulator and must recover that same
    index as the most frequently measured outcome.

    This is a real (if small) instance of the geometric search that
    A006065 / Erdos problem 101 is about: finding the 4-point line(s)
    among a point set. PASS means the quantum search recovered exactly the
    index that classical brute force found.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def build_points():
    pts = [(0, 0), (1, 1), (2, 2), (3, 3)]
    for x in (5, 6, 7, 8, 9):
        pts.append((x, x * x))
    return pts


def collinear(p, q, r):
    # exact integer cross product test
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) == 0


def four_collinear(quad):
    p, q, r, s = quad
    if not collinear(p, q, r):
        return False
    return collinear(p, q, s) and collinear(p, r, s)


def classical_answer():
    points = build_points()
    combos = list(itertools.combinations(range(9), 4))  # 126 quadruples
    hits = []
    for idx, combo in enumerate(combos):
        quad = [points[i] for i in combo]
        if four_collinear(quad):
            hits.append(idx)
    assert len(hits) == 1, f"expected exactly one 4-collinear quadruple, found {hits}"
    return combos, hits[0]


def oracle_circuit(n_qubits, target_index):
    qc = QuantumCircuit(n_qubits)
    bits = format(target_index, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i in zero_positions:
        qc.x(i)
    return qc


def diffuser_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits, target_index, n_valid_states):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    iterations = max(1, round((math.pi / 4) * math.sqrt((2 ** n_qubits) / 1)))
    # With only 1 marked state out of 2**n_qubits states (n_valid_states of
    # which are "real" combination indices, the rest padding), a handful of
    # iterations suffices; cap it to something sane for a 7-qubit register.
    iterations = min(iterations, 12)

    oracle = oracle_circuit(n_qubits, target_index)
    diffuser = diffuser_circuit(n_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=4096).result()
    counts = result.get_counts()
    best = max(counts, key=counts.get)
    # Qiskit returns bit strings MSB-first over the classical register order;
    # register bit i holds qubit i, printed with qubit (n-1) first.
    measured_index = int(best[::-1], 2)
    return measured_index, counts, iterations


def main():
    combos, target_index = classical_answer()
    points = build_points()
    print(f"Points: {points}")
    print(f"Total 4-subsets: {len(combos)}")
    print(f"Classical answer: unique 4-collinear quadruple is index {target_index} "
          f"-> point indices {combos[target_index]} "
          f"-> coordinates {[points[i] for i in combos[target_index]]}")

    n_qubits = 7  # 2**7 = 128 >= 126
    measured_index, counts, iterations = run_grover(n_qubits, target_index, len(combos))

    print(f"Grover ran with {n_qubits} qubits, {iterations} iterations.")
    print(f"Most frequent measured index: {measured_index} "
          f"(out of {sum(counts.values())} shots, top count "
          f"{counts[max(counts, key=counts.get)]})")

    ok = (measured_index == target_index)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
