"""
Erdos problem #700 -- quantum-testable instance of OEIS A091963.

Erdos problem #700 (erdosproblems.com, tags: number theory, binomial
coefficients) is linked in the local problems.yaml record to OEIS sequence
A091963: "a(n) is the smallest gcd of two interior numbers on row n of
Pascal's triangle ('interior' means that the 1's at the ends of the rows are
excluded)."  I.e. for row n, look at the interior binomial coefficients
C(n,1), C(n,2), ..., C(n,n-1); a(n) is the minimum of gcd(C(n,i), C(n,j))
taken over all pairs 1 <= i < j <= n-1.

The known terms of A091963 (indexed starting at n=2) are:
2, 3, 2, 5, 2, 7, 2, 3, 2, 11, ...   (n = 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...)
so a(6) = 2.

Classical property tested here
-------------------------------
Fix n = 6.  The interior row is:
    C(6,1)=6, C(6,2)=15, C(6,3)=20, C(6,4)=15, C(6,5)=6
There are C(5,2) = 10 unordered pairs (i,j), 1<=i<j<=5. Enumerate them in a
fixed order (index 0..9) and compute gcd(C(6,i), C(6,j)) for every pair,
directly from first principles (no OEIS values copied) using Python's
math.comb and math.gcd. The minimum gcd over all 10 pairs is the classical
value of a(6), and the set of pair-indices that attain this minimum is the
"marked" set for a Grover search.

Quantum circuit
----------------
This is turned into a genuine (if small) Grover search: encode the 10 pair
indices as 4-qubit basis states 0..9 (basis states 10..15 are unused /
never marked), build an oracle that phase-flips exactly the basis states
whose pair achieves the classical minimum gcd, and run standard Grover
amplitude amplification (with the optimal number of iterations for this
search-space size) on the ideal AerSimulator. The circuit is expected to
return, with high probability, one of the marked pair-indices.

PASS/FAIL is determined by checking that the most frequent measurement
outcome(s) in the sampled counts correspond to a pair index whose gcd
(recomputed classically) equals the classically-computed minimum -- i.e.
the quantum search actually found a true witness of a(6).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_solution():
    """Compute a(6) for A091963 from first principles, and the marked pairs."""
    n = 6
    interior = [math.comb(n, k) for k in range(1, n)]  # C(6,1..5)
    pairs = list(combinations(range(len(interior)), 2))  # 10 pairs, fixed order
    gcds = [math.gcd(interior[i], interior[j]) for (i, j) in pairs]
    min_gcd = min(gcds)
    marked_indices = [idx for idx, g in enumerate(gcds) if g == min_gcd]
    return interior, pairs, gcds, min_gcd, marked_indices


def build_oracle(num_qubits, marked_indices):
    """Phase-flip oracle marking each index in marked_indices (as a 4-bit state)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{num_qubits}b")[::-1]  # little-endian per qubit order
        # Flip qubits that should be 0 so the target pattern becomes all-1s
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # Multi-controlled Z on all qubits (phase flip when all qubits are 1)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(marked_indices, search_space_size=10, num_qubits=4, shots=4096):
    # Optimal number of Grover iterations for this |marked| / N ratio.
    num_marked = len(marked_indices)
    theta = math.asin(math.sqrt(num_marked / (2 ** num_qubits)))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(num_qubits, marked_indices)
    diffuser = build_diffuser(num_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(num_qubits))
        qc.append(diffuser.to_instruction(), range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    interior, pairs, gcds, min_gcd, marked_indices = classical_solution()

    print("Erdos problem #700 / OEIS A091963")
    print(f"n = 6, interior row C(6,1..5) = {interior}")
    print(f"pairs (index -> (i,j), gcd): "
          f"{[(idx, pairs[idx], gcds[idx]) for idx in range(len(pairs))]}")
    print(f"classical min gcd (a(6)) = {min_gcd}")
    print(f"marked pair-indices achieving the minimum: {marked_indices}")

    counts, iterations = run_grover(marked_indices)
    print(f"Grover iterations used: {iterations}")
    print(f"measurement counts: {counts}")

    # Restrict to the valid pair-index range (0..9); states 10..15 are unused.
    valid_counts = {
        state: c for state, c in counts.items()
        if int(state, 2) < len(pairs)
    }
    total_shots = sum(counts.values())
    valid_shots = sum(valid_counts.values())

    # Top measured outcome among valid pair-indices.
    top_state = max(valid_counts, key=valid_counts.get)
    top_index = int(top_state, 2)
    top_prob = valid_counts[top_state] / total_shots

    # Verify: the top outcome must be one of the classically marked indices,
    # and must correspond to a pair whose gcd equals the classical minimum,
    # and Grover amplification must have concentrated a clear majority of
    # probability on marked outcomes.
    marked_shots = sum(c for s, c in counts.items() if int(s, 2) in marked_indices)
    marked_prob = marked_shots / total_shots

    top_is_marked = top_index in marked_indices
    top_gcd_matches = (top_index < len(pairs)) and (gcds[top_index] == min_gcd)
    amplification_ok = marked_prob > 0.5  # much better than uniform 10/16 ~ 0.625...
    # note: with only 10 valid/16 states uniform baseline for a marked draw
    # would be num_marked/16; require clear amplification above that baseline.
    uniform_baseline = len(marked_indices) / (2 ** 4)
    amplification_ok = marked_prob > 2 * uniform_baseline

    passed = top_is_marked and top_gcd_matches and amplification_ok

    print(f"top measured pair-index: {top_index} (prob {top_prob:.3f}), "
          f"gcd={gcds[top_index] if top_index < len(pairs) else 'invalid'}")
    print(f"probability mass on marked (classically correct) outcomes: "
          f"{marked_prob:.3f} (uniform baseline {uniform_baseline:.3f})")

    if passed:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
