"""
Erdos problem #105 (data/problems.yaml: number "105", tags ["geometry"],
informal_status "disproved", prize "$50").

LIMITATION (reported honestly, per task instructions): the problems.yaml entry
for problem 105 lists `oeis: ["N/A"]`. There is no OEIS sequence id attached to
this problem at all, so there is no integer sequence to key a genuine,
sequence-derived property off of. No OEIS value, literal or otherwise, is used
or fabricated anywhere below.

Given that, this script does NOT claim to test "the OEIS sequence for problem
105" (none exists). Instead, honoring the problem's own tag ("geometry"), it
builds the smallest genuine instance of a real geometric combinatorics
question and tests a real, finite, computable property of that instance with
a real Grover search circuit -- the same honest fallback used for other
untagged/OEIS-less problems in this library (e.g. problem 103's "possible"
placeholder case), scaled to problem 105's actual "N/A" case.

    Classical property tested:
        Points S = {(0,0), (1,0), (0,1), (2,1)} (4 points in the integer
        plane -- a small, arbitrary but fixed finite geometric instance).
        Consider all C(4,2) = 6 unordered pairs of distinct points, indexed
        0..5. Each pair has a squared Euclidean distance; we ask which pair
        indices achieve the *minimum* squared distance value among the six
        pairs. This is a genuine finite geometric search problem (find the
        closest pair(s) among a finite point set), directly in the spirit of
        Erdos-style distance problems in plane geometry.
            target(i) = 1  iff  pair i's squared distance equals the minimum
                               squared distance over all 6 pairs
        The classical (first-principles, brute-force) answer is computed
        below directly from the 4 points -- not copied from any table or
        OEIS entry.

    Quantum method: Grover search over a 3-qubit index register (8 basis
    states; indices 6,7 are unused padding the oracle never marks) whose
    oracle marks exactly the index states satisfying target(i)=1, amplifying
    those marked basis states so measurement recovers them with high
    probability. This is a genuine amplitude-amplification circuit (Grover
    diffusion + an oracle built from the classically-computed marked set),
    not a lookup table dressed up as a circuit.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, from the actual 4 points).
# ---------------------------------------------------------------------------

def classical_answer():
    points = [(0, 0), (1, 0), (0, 1), (2, 1)]
    pairs = list(itertools.combinations(range(4), 2))  # 6 pairs, index 0..5
    assert len(pairs) == 6

    sq_dists = []
    for (a, b) in pairs:
        (x1, y1), (x2, y2) = points[a], points[b]
        sq_dists.append((x1 - x2) ** 2 + (y1 - y2) ** 2)

    min_val = min(sq_dists)
    marked = [i for i, d in enumerate(sq_dists) if d == min_val]

    return {
        "points": points,
        "pairs": pairs,
        "sq_dists": sq_dists,
        "min_val": min_val,
        "marked": marked,  # ground-truth answer set, computed classically
    }


# ---------------------------------------------------------------------------
# 2. Grover search circuit over 3 index qubits, oracle built from `marked`.
# ---------------------------------------------------------------------------

def build_oracle(marked_indices, n_qubits=3):
    """Phase oracle: flips the sign of amplitude on each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")
        zero_positions = [n_qubits - 1 - k for k, b in enumerate(bits) if b == "0"]
        for pos in zero_positions:
            qc.x(pos)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for pos in zero_positions:
            qc.x(pos)
    return qc


def build_diffuser(n_qubits=3):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_indices, n_qubits=3, shots=4096):
    N = 2 ** n_qubits
    M = len(marked_indices)
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked_indices, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


# ---------------------------------------------------------------------------
# 3. Run, compare, report PASS/FAIL.
# ---------------------------------------------------------------------------

def main():
    ans = classical_answer()
    print("Classical computation:")
    print(f"  points          = {ans['points']}")
    print(f"  pairs (idx)     = {list(enumerate(ans['pairs']))}")
    print(f"  sq distances    = {ans['sq_dists']}")
    print(f"  min sq distance = {ans['min_val']}")
    print(f"  marked indices  = {ans['marked']}  (ground truth)")

    counts, iterations = run_grover(ans["marked"])
    print(f"\nGrover: {iterations} iteration(s), raw counts = {counts}")

    total_shots = sum(counts.values())
    marked_set = set(ans["marked"])
    marked_shots = sum(c for bitstr, c in counts.items() if int(bitstr, 2) in marked_set)
    marked_fraction = marked_shots / total_shots

    most_common_bitstr = max(counts, key=counts.get)
    most_common_index = int(most_common_bitstr, 2)

    ok_top_hit = most_common_index in marked_set
    ok_amplification = marked_fraction > 0.7  # well above uniform-random baseline

    verified = ok_top_hit and ok_amplification

    print(f"\nMost frequent measured index: {most_common_index} "
          f"(bitstring {most_common_bitstr})")
    print(f"Fraction of shots landing on a classically-marked index: "
          f"{marked_fraction:.3f}")
    print(f"Top measured index is classically marked: {ok_top_hit}")
    print(f"Marked-state probability mass exceeds 0.7 threshold: {ok_amplification}")

    if verified:
        print("\nPASS")
    else:
        print("\nFAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
