"""
Erdos problem #670 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
`number: "670"`):
    prize: no
    status: open
    oeis: ["N/A"]
    tags: ["geometry", "distances"]

HONESTY NOTE ON LIMITATION
---------------------------
Problem #670 has **no OEIS sequence id** attached in the source data
(`oeis: ["N/A"]`). There is therefore no actual integer sequence from OEIS
to build a "sequence membership" oracle around, and this script does not
pretend otherwise. What the metadata does give us is a real, specific
mathematical *topic*: the tags are "geometry" and "distances", which is
Erdos's classic area of distinct-distances-in-point-sets questions (e.g.
the Erdos distinct distances problem: how few distinct pairwise distances
can a set of n points in the plane determine?).

So instead of fabricating or mis-citing an OEIS value, this script builds
a genuine finite, computable instance *in the same mathematical spirit as
the problem's tags*: given a fixed base triangle of 3 points and a small
finite list of 8 integer-coordinate candidate points, which candidate
point, when added to the triangle, yields the FEWEST distinct pairwise
distances among the resulting 4-point set? This is a real combinatorial
search problem (distinct-distance minimization) over a small, exactly
computable search space of size 8 (3 qubits) -- squarely a Grover-search
instance, computed and checked classically first, then verified by a real
Grover circuit on AerSimulator.

Classical answer for this instance is computed from first principles
below (function `distinct_distance_count`) and printed before the quantum
run, so nothing is asserted without derivation.

Because the underlying Erdos problem itself has no OEIS id, this is
reported as ran_ok=True / verified_against_classical=True for the
constructed proxy instance, but NOT as a genuine OEIS-sequence circuit --
that part of the assignment could not be honestly fulfilled for #670.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical setup: fixed base triangle + 8 candidate 4th points.
# ---------------------------------------------------------------------------

BASE_TRIANGLE = [(0, 0), (4, 0), (0, 3)]  # a 3-4-5 right triangle

# 8 candidate points for the 4th vertex, addressed by a 3-bit index 0..7.
CANDIDATES = [
    (1, 1),
    (2, 2),
    (4, 3),
    (3, 4),
    (-1, 0),
    (0, -1),
    (2, 0),
    (0, 5),
]


def sq_dist(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def distinct_distance_count(points):
    """Number of distinct pairwise (squared) distances among `points`."""
    dists = {sq_dist(p, q) for p, q in combinations(points, 2)}
    return len(dists)


def classical_best_index():
    """Brute-force the candidate index minimizing distinct-distance count."""
    counts = []
    for cand in CANDIDATES:
        cnt = distinct_distance_count(BASE_TRIANGLE + [cand])
        counts.append(cnt)
    best = min(range(len(CANDIDATES)), key=lambda i: counts[i])
    # Require a UNIQUE minimizer so the Grover oracle marks exactly one state.
    assert counts.count(counts[best]) == 1, (
        "instance is degenerate (non-unique minimizer); "
        f"counts={counts}"
    )
    return best, counts


TARGET_INDEX, ALL_COUNTS = classical_best_index()
N_QUBITS = 3  # log2(8 candidates)

print("Base triangle:", BASE_TRIANGLE)
print("Candidate points (index -> point -> distinct-distance count):")
for i, (cand, cnt) in enumerate(zip(CANDIDATES, ALL_COUNTS)):
    marker = "  <-- classical minimum" if i == TARGET_INDEX else ""
    print(f"  {i:03b} : {cand} -> {cnt}{marker}")
print(f"Classical target index (unique minimizer): {TARGET_INDEX} "
      f"({TARGET_INDEX:03b})")


# ---------------------------------------------------------------------------
# 2. Grover search circuit that finds TARGET_INDEX among 8 basis states.
# ---------------------------------------------------------------------------

def build_oracle(n_qubits, target_index):
    """Phase-flip the |target_index> basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(target_index, f"0{n_qubits}b")
    # Flip qubits that should be 0 in the target so the controlled-Z fires
    # exactly on |target_index>.
    for i, b in enumerate(reversed(bits)):
        if b == "0":
            qc.x(i)
    if n_qubits == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
        qc.append(mcz, list(range(n_qubits)))
    for i, b in enumerate(reversed(bits)):
        if b == "0":
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
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


def grover_circuit(n_qubits, target_index):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, target_index)
    diffuser = build_diffuser(n_qubits)

    n_items = 2 ** n_qubits
    n_iterations = max(1, round(math.pi / 4 * math.sqrt(n_items)))
    for _ in range(n_iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, n_iterations


def run_grover():
    qc, n_iter = grover_circuit(N_QUBITS, TARGET_INDEX)
    backend = AerSimulator()
    shots = 2000
    transpiled = qc.decompose(reps=5)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports bit strings MSB..LSB matching qubit order q_{n-1}...q_0;
    # our index encoding used qubit i as bit i (LSB = qubit 0), matching the
    # classical `format(target_index, '0{n}b')` used with `reversed()` above,
    # so the returned bitstring (as-is) equals that same binary form.
    most_common = max(counts, key=counts.get)
    measured_index = int(most_common, 2)
    success_prob = counts.get(most_common, 0) / shots

    print(f"\nGrover iterations used: {n_iter}")
    print(f"Measurement histogram (top 5): "
          f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
    print(f"Most frequent measured index: {measured_index} "
          f"({most_common}), probability ~{success_prob:.3f}")

    return measured_index, success_prob


if __name__ == "__main__":
    measured_index, success_prob = run_grover()

    ok = (measured_index == TARGET_INDEX) and (success_prob > 0.5)

    print(f"\nClassical answer : index {TARGET_INDEX}")
    print(f"Quantum answer   : index {measured_index} "
          f"(probability {success_prob:.3f})")

    if ok:
        print("PASS")
    else:
        print("FAIL")
