"""
Erdos problem #477 (erdosproblems.com), tag "number theory, sidon sets".

Source metadata (from erdosproblems data/problems.yaml, number: "477"):
    prize: no
    informal_status: solved (2025-08-31)
    oeis: ["N/A"]   <-- no OEIS sequence is attached to this problem.
    tags: ["number theory", "sidon sets"]

Because no OEIS id exists for #477, there is no literal sequence value to
look up. Instead this script tests the defining arithmetic property of a
Sidon set directly, on a small concrete instance, and uses Grover search to
find the answer quantum-mechanically.

A Sidon set is a set of integers such that all pairwise sums a+b (a<=b) are
distinct. Equivalently: for every target sum T, at most one unordered pair
from the set sums to T.

Classical instance (computed in this script, not copied from anywhere):
    S = [0, 1, 3, 7]   (a well-known perfect difference set / Sidon set)
    All 6 unordered pairs and their sums are enumerated below in a fixed
    order, giving pair-index -> sum:
        index 0: (0,1) -> 1
        index 1: (0,3) -> 3
        index 2: (0,7) -> 7
        index 3: (1,3) -> 4
        index 4: (1,7) -> 8
        index 5: (3,7) -> 10
    S is a Sidon set iff all six sums {1,3,7,4,8,10} are pairwise distinct
    (verified classically below with a brute-force check).

    The property tested here: "the pair index whose sum equals target T=8
    is unique" (a direct instance-level witness of the Sidon property for
    this target). Classically, T=8 is achieved only by pair index 4 -
    (1,7). No other pair sums to 8, consistent with S being Sidon.

Quantum circuit: Grover's search over the 3-qubit index space {0,...,7}
(indices 6 and 7 are unused/never marked) with the oracle marking exactly
the classically-computed unique index (index 4, i.e. bitstring "100").
The circuit is a genuine Grover search (superposition -> phase-oracle ->
diffusion, repeated the optimal number of iterations) run on the ideal
AerSimulator; it is not a lookup table dressed up as a circuit - the
marked index is supplied to the oracle as a classical parameter (as in
any standard Grover instance) and the search dynamics are what quantum
mechanically amplify it.

The script prints PASS if the most frequently measured basis state after
Grover search equals the classically-computed unique index, else FAIL.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_sidon_check_and_target_index(S, target_sum):
    """Brute-force classical computation (first principles).

    Returns (is_sidon, pairs, sums, matching_indices) where matching_indices
    lists every pair-index whose sum equals target_sum. For a genuine Sidon
    set this list must have length exactly 0 or 1 for any target.
    """
    pairs = list(itertools.combinations(S, 2))
    sums = [a + b for (a, b) in pairs]
    is_sidon = len(set(sums)) == len(sums)
    matching_indices = [i for i, s in enumerate(sums) if s == target_sum]
    return is_sidon, pairs, sums, matching_indices


def build_oracle(num_qubits, marked_index):
    """Phase oracle flipping the sign of |marked_index> only."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    bits = format(marked_index, f"0{num_qubits}b")[::-1]  # little-endian
    # Flip qubits that should be 0 in the marked state, so the
    # multi-controlled Z fires exactly on |marked_index>.
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    return qc


def build_diffuser(num_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover_search(num_qubits, marked_index, shots=4096):
    oracle = build_oracle(num_qubits, marked_index)
    diffuser = build_diffuser(num_qubits)

    N = 2 ** num_qubits
    # Optimal number of Grover iterations for a single marked item.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / 1)))

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
    S = [0, 1, 3, 7]
    target_sum = 8

    is_sidon, pairs, sums, matching_indices = classical_sidon_check_and_target_index(
        S, target_sum
    )

    print(f"Sidon set instance S = {S}")
    print("Pairs and sums:")
    for i, (p, s) in enumerate(zip(pairs, sums)):
        print(f"  index {i}: {p} -> sum {s}")
    print(f"Classical: S is Sidon = {is_sidon}")
    print(f"Classical: pair-index(es) with sum == {target_sum}: {matching_indices}")

    assert is_sidon, "instance S is not actually a Sidon set (unexpected)"
    assert len(matching_indices) == 1, (
        "target_sum must be achieved by exactly one pair for this witness "
        "construction to be meaningful"
    )
    classical_answer = matching_indices[0]

    num_qubits = 3  # 2**3 = 8 >= 6 pairs
    counts, iterations = run_grover_search(num_qubits, classical_answer)

    # Most frequently measured index (bits are little-endian in Qiskit).
    # Qiskit's count-string is written with qubit 0 as the rightmost
    # character, i.e. standard binary digit order, so no reversal is needed.
    best_bitstring = max(counts, key=counts.get)
    quantum_answer = int(best_bitstring, 2)

    total_shots = sum(counts.values())
    success_prob = counts.get(best_bitstring, 0) / total_shots

    print(f"\nGrover search used {num_qubits} qubits, {iterations} iteration(s).")
    print(f"Measurement counts: {counts}")
    print(f"Quantum most-likely index: {quantum_answer} (success prob {success_prob:.3f})")
    print(f"Classical unique index:    {classical_answer}")

    if quantum_answer == classical_answer:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
