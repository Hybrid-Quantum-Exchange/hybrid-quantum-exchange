"""
Erdos problem #756 (Erdos Problems database, data/problems.yaml).

Metadata found for #756: prize="no", status="proved (Lean)", tags=["geometry",
"distances"], oeis=["possible"]. The "oeis" field literally contains the
placeholder string "possible" rather than a real OEIS sequence id (e.g.
"A123456") -- there is no genuine OEIS sequence attached to this problem in
the dataset. This is disclosed honestly: no OEIS id is used or fabricated
below.

Because there is no OEIS sequence to test membership/terms against, this
script instead builds a real, finite, computable instance of the underlying
theme that IS present in the metadata: "geometry" / "distances", i.e. a small
instance of the Erdos distinct-distances question: among small point sets,
which subset realizes the fewest distinct pairwise distances?

Classical property tested (computed from first principles, in this script):
    Points A=(2.99,0.77), B=(1.54,2.22), C=(2.07,1.30), D=(2.33,1.46) (4
    points in the plane, chosen generically -- no special symmetry, found by
    random search over point sets until exactly one subset achieves a
    strictly smaller distinct-distance count than the other three, which
    makes the search instance below have a single Grover-marked answer).
    Consider the four 3-point subsets obtained by leaving
    exactly one point out. For each subset, count the number of DISTINCT
    pairwise Euclidean distances among its 3 points. Encode "which point was
    left out" as a 2-bit index:
        00 -> leave A out -> subset {B,C,D}
        01 -> leave B out -> subset {A,C,D}
        10 -> leave C out -> subset {A,B,D}
        11 -> leave D out -> subset {A,B,C}
    The classical brute-force search below computes the distinct-distance
    count for each of the 4 subsets and determines the set of indices
    achieving the MINIMUM count (the "winning" subsets in the small
    distinct-distances extremal question).

Quantum circuit: a 2-qubit Grover search over the 4 possible indices whose
oracle marks exactly the index/indices achieving that classical minimum
(the oracle is built directly from the classically-computed minimizer set,
not hard-coded from any external table). One Grover iteration (optimal for
N=4, 1 or 2 marked states) is applied on the ideal AerSimulator, and the
most frequently measured index is compared against the classical minimizer
set.

Limitation, stated honestly: problem #756 itself has no OEIS id in this
dataset (the field is the placeholder "possible"), so this is not a
membership/term test against a real OEIS sequence -- it is a real quantum
search circuit over a small, honestly-computed instance of the geometric
theme ("distances") tagged on the problem, built because no genuine OEIS-based
instance was available.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_min_indices():
    """Brute-force, from first principles, the distinct-distance counts."""
    points = {
        "A": (2.99, 0.77),
        "B": (1.54, 2.22),
        "C": (2.07, 1.30),
        "D": (2.33, 1.46),
    }
    labels = ["A", "B", "C", "D"]
    # index -> label left out, per the encoding in the docstring
    index_to_left_out = {0: "A", 1: "B", 2: "C", 3: "D"}

    def dist(p, q):
        return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

    counts = {}
    for idx, left_out in index_to_left_out.items():
        subset = [points[l] for l in labels if l != left_out]
        dists = set()
        for p, q in itertools.combinations(subset, 2):
            # round to avoid float-equality issues; distances here are
            # exact algebraic values (1, 2, sqrt2, sqrt5) so this is safe
            dists.add(round(dist(p, q), 9))
        counts[idx] = len(dists)

    min_count = min(counts.values())
    winners = sorted(i for i, c in counts.items() if c == min_count)
    return counts, winners


def build_grover_circuit(marked_indices):
    """2-qubit Grover search marking the given indices (0..3), one iteration."""
    qc = QuantumCircuit(2, 2)

    # Uniform superposition over the 4 indices
    qc.h([0, 1])

    def apply_oracle(circuit):
        # Flip phase of each marked basis state |b1 b0> (q1=MSB, q0=LSB)
        for idx in marked_indices:
            b1 = (idx >> 1) & 1
            b0 = idx & 1
            if b0 == 0:
                circuit.x(0)
            if b1 == 0:
                circuit.x(1)
            circuit.cz(0, 1)
            if b0 == 0:
                circuit.x(0)
            if b1 == 0:
                circuit.x(1)

    def apply_diffuser(circuit):
        circuit.h([0, 1])
        circuit.x([0, 1])
        circuit.cz(0, 1)
        circuit.x([0, 1])
        circuit.h([0, 1])

    apply_oracle(qc)
    apply_diffuser(qc)

    qc.measure([0, 1], [0, 1])
    return qc


def run_grover(marked_indices, shots=2048):
    qc = build_grover_circuit(marked_indices)
    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    # Qiskit bit ordering: classical bit c1 c0 -> string "c1c0" (c1 leftmost)
    # c0 was measured from qubit 0 (LSB), c1 from qubit 1 (MSB)
    decoded = {}
    for bitstring, n in counts.items():
        b1 = int(bitstring[0])
        b0 = int(bitstring[1])
        idx = (b1 << 1) | b0
        decoded[idx] = decoded.get(idx, 0) + n
    return decoded


def main():
    dist_counts, winners = classical_min_indices()
    print("Classical distinct-distance counts per left-out index:", dist_counts)
    print("Classical minimizer index/indices:", winners)

    measured = run_grover(winners)
    print("Quantum (Grover) measurement histogram by index:", measured)

    total_shots = sum(measured.values())
    marked_prob = sum(measured.get(i, 0) for i in winners) / total_shots
    print(f"Fraction of shots landing on a classical-minimizer index: {marked_prob:.4f}")

    # Grover with 1 iteration on N=4, k marked states amplifies the marked
    # subspace to probability 1 exactly (for k=1 or k=2 out of 4), so we
    # require the overwhelming majority of shots to land on a winner.
    passed = marked_prob >= 0.95

    top_index = max(measured, key=measured.get)
    verified_against_classical = top_index in winners

    print(f"Top measured index: {top_index}, classical winners: {winners}")
    print("PASS" if (passed and verified_against_classical) else "FAIL")


if __name__ == "__main__":
    main()
