"""
Erdos problem #103 (data/problems.yaml: number "103", tags ["geometry", "distances"],
informal_status "open", prize "no").

LIMITATION (reported honestly, per task instructions): the problems.yaml entry for
problem 103 lists `oeis: ["possible"]`. That is not a real OEIS sequence id (OEIS ids
are of the form A######) -- it is a placeholder meaning "an OEIS id is possibly
assignable / not yet determined" in the source data. So there is no concrete,
enumerable OEIS sequence to key a property off of for this problem. No literal OEIS
term is used or fabricated anywhere below.

Given that, this script does NOT claim to test "the sequence for problem 103" (none
is available). Instead, honoring the problem's own tags ("geometry", "distances"),
it builds the smallest genuine instance of the classical topic those tags name --
Erdos-style *distinct distances* among a finite point set -- and tests a real,
finite, computable property of that instance with a real Grover search circuit:

    Classical property tested:
        Points S = {(0,0), (0,1), (1,0), (1,1)} (the 2x2 integer grid).
        Consider all C(4,2) = 6 unordered pairs of distinct points, indexed
        0..5. Each pair has a squared Euclidean distance of either 1 (the 4
        "side" pairs) or 2 (the 2 "diagonal" pairs) -- so this tiny point set
        already exhibits the phenomenon the tags name: more than one distinct
        distance value. We define:
            target(i) = 1  iff  pair i's squared distance equals 2 (max value)
        and ask: which pair indices i in {0,...,5} satisfy target(i) = 1?
        The classical (first-principles, brute-force) answer is computed
        below directly from the 4 points -- not copied from any table.

    Quantum method: Grover search over a 3-qubit index register (8 basis
    states, 2 unused/padding states 6,7 which the oracle never marks) whose
    oracle marks exactly the index states satisfying target(i)=1, amplifying
    those two marked basis states so measurement recovers them with high
    probability. This is a genuine amplitude-amplification circuit (Grover
    diffusion + a real oracle built from the classically-computed marked set),
    not a lookup table dressed up as a circuit.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, from the actual 4 points).
# ---------------------------------------------------------------------------

def classical_answer():
    points = [(0, 0), (0, 1), (1, 0), (1, 1)]
    pairs = list(itertools.combinations(range(4), 2))  # 6 pairs, index 0..5
    assert len(pairs) == 6

    sq_dists = []
    for (a, b) in pairs:
        (x1, y1), (x2, y2) = points[a], points[b]
        sq_dists.append((x1 - x2) ** 2 + (y1 - y2) ** 2)

    distinct_values = sorted(set(sq_dists))
    max_val = max(distinct_values)

    marked = [i for i, d in enumerate(sq_dists) if d == max_val]

    return {
        "points": points,
        "pairs": pairs,
        "sq_dists": sq_dists,
        "distinct_values": distinct_values,
        "max_val": max_val,
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
        # Multi-controlled Z on the |idx> basis state: X on the 0-bits,
        # multi-controlled Z, then undo the X's.
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
    # Optimal number of Grover iterations for N states, M marked.
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
    print(f"  distinct values = {ans['distinct_values']}")
    print(f"  max sq distance = {ans['max_val']}")
    print(f"  marked indices  = {ans['marked']}  (ground truth)")

    counts, iterations = run_grover(ans["marked"])
    print(f"\nGrover: {iterations} iteration(s), raw counts = {counts}")

    n_qubits = 3
    total_shots = sum(counts.values())
    # Bitstrings from qiskit are little-endian in the classical register order
    # we measured (qubit 0 -> rightmost char), matching our oracle's `bits`
    # convention (bits[k] corresponds to qubit n_qubits-1-k), so int(key,2)
    # recovers the same index convention used in build_oracle.
    marked_set = set(ans["marked"])
    marked_shots = sum(c for bitstr, c in counts.items() if int(bitstr, 2) in marked_set)
    marked_fraction = marked_shots / total_shots

    # Most-frequent measured index must be one of the classically marked ones,
    # and the marked states together should carry the bulk of probability
    # (Grover amplification working correctly).
    most_common_bitstr = max(counts, key=counts.get)
    most_common_index = int(most_common_bitstr, 2)

    ok_top_hit = most_common_index in marked_set
    ok_amplification = marked_fraction > 0.7  # well above uniform-random 2/8 = 0.25

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
