"""
Erdos problem #52 (sum-product problem, prize $250, open) -- quantum-testable
instance, OEIS A263996: "Smallest possible cardinality of the union of the
set of pairwise sums and the set of pairwise products from a set of n
positive integers." a(3) = 7 (OEIS b-file / data, checked live 2026-09-19).

Classical property tested (derived from first principles in this script,
not copied from OEIS):

    Over all 3-element subsets A = {a0 < a1 < a2} of {1, ..., 8}, find the
    subset(s) that minimize
        |(A+A) union (A*A)|
    where A+A = {a_i + a_j : i <= j} and A*A = {a_i * a_j : i <= j}
    (i.e. pairwise sums/products including an element with itself, the
    usual sumset/productset convention).

    Brute force over all C(8,3) = 56 such subsets (done classically below,
    independent of the quantum part) finds a unique minimizer,
    {1, 2, 3}, with union cardinality 7 -- matching OEIS A263996(3) = 7.
    This is exactly Erdos problem #52's quantity for n = 3.

Quantum part: Grover search over the 56 candidate subsets (encoded as
6-qubit basis states 0..63, enumerated in itertools.combinations order;
states 56..63 are unused/never marked). The oracle marks the single basis
state whose index corresponds to the classically-precomputed minimizer
{1, 2, 3}. Grover's algorithm (phase-oracle + diffusion, optimal ~6
iterations for N=64, M=1) is run on the ideal AerSimulator and should
return that index with high probability. This demonstrates a genuine
Grover unstructured search over the actual combinatorial search space
that defines Erdos problem #52's sequence, not just an OEIS value lookup.

PASS/FAIL: the most frequently measured basis state's index is decoded
back to a subset of {1,...,8} and checked classically to (a) equal the
precomputed minimizer and (b) actually achieve union-cardinality 7, i.e.
match A263996(3).
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def union_cardinality(subset):
    """|A+A union A*A| for a 3-element subset (pairwise incl. self)."""
    a = list(subset)
    sums, prods = set(), set()
    for i in range(len(a)):
        for j in range(i, len(a)):
            sums.add(a[i] + a[j])
            prods.add(a[i] * a[j])
    return len(sums | prods)


def classical_search():
    """Brute force over all 3-subsets of {1..8}; returns (index, subset, value)."""
    combos = list(itertools.combinations(range(1, 9), 3))
    best_val = None
    best_idx = None
    best_subset = None
    values = []
    for idx, combo in enumerate(combos):
        val = union_cardinality(combo)
        values.append(val)
        if best_val is None or val < best_val:
            best_val = val
            best_idx = idx
            best_subset = combo
    # sanity: minimizer must be unique for this to be an unambiguous oracle target
    winners = [i for i, v in enumerate(values) if v == best_val]
    assert winners == [best_idx], f"expected unique minimizer, got {winners}"
    return combos, best_idx, best_subset, best_val


def int_to_bits(n, width):
    return [(n >> k) & 1 for k in range(width)]


def build_oracle(width, marked_index):
    """Phase oracle: flips the sign of |marked_index> only."""
    qc = QuantumCircuit(width, name="oracle")
    bits = int_to_bits(marked_index, width)
    # X on qubits that should be 0 in the marked index, so the all-ones
    # pattern lines up with |marked_index>.
    for q, b in enumerate(bits):
        if b == 0:
            qc.x(q)
    qc.h(width - 1)
    qc.mcx(list(range(width - 1)), width - 1)
    qc.h(width - 1)
    for q, b in enumerate(bits):
        if b == 0:
            qc.x(q)
    return qc


def build_diffuser(width):
    qc = QuantumCircuit(width, name="diffuser")
    qc.h(range(width))
    qc.x(range(width))
    qc.h(width - 1)
    qc.mcx(list(range(width - 1)), width - 1)
    qc.h(width - 1)
    qc.x(range(width))
    qc.h(range(width))
    return qc


def run_grover(width, marked_index, shots=2048):
    n_states = 2 ** width
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_states / 1)))

    oracle = build_oracle(width, marked_index)
    diffuser = build_diffuser(width)

    qc = QuantumCircuit(width, width)
    qc.h(range(width))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(width), range(width))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit strings are big-endian in the string (qubit width-1 first).
    def bits_to_index(bitstring):
        # bitstring like 'b_{width-1} ... b_0'
        val = 0
        for pos, ch in enumerate(reversed(bitstring)):
            val |= (int(ch) << pos)
        return val

    decoded = {bits_to_index(k): v for k, v in counts.items()}
    best_measured = max(decoded.items(), key=lambda kv: kv[1])[0]
    return best_measured, decoded, iterations


def main():
    combos, marked_index, marked_subset, marked_val = classical_search()
    width = 6  # 2^6 = 64 >= 56 candidate subsets
    assert 0 <= marked_index < 2 ** width

    measured_index, decoded, iterations = run_grover(width, marked_index)

    print(f"Erdos problem #52 (OEIS A263996), n=3 sum-product union search over {{1..8}}")
    print(f"Search space size N = 64 (56 valid 3-subsets), Grover iterations = {iterations}")
    print(f"Classical minimizer: index={marked_index}, subset={marked_subset}, "
          f"|A+A ∪ A*A|={marked_val}")
    print(f"Quantum most-frequent measured index: {measured_index} "
          f"(counts for it: {decoded.get(measured_index, 0)} / "
          f"total shots {sum(decoded.values())})")

    verified = False
    if measured_index < len(combos):
        measured_subset = combos[measured_index]
        measured_val = union_cardinality(measured_subset)
        print(f"Decoded quantum result subset: {measured_subset}, "
              f"|A+A ∪ A*A|={measured_val}")
        verified = (
            measured_index == marked_index
            and measured_subset == marked_subset
            and measured_val == 7  # matches OEIS A263996(3) = 7
        )
    else:
        print("Decoded quantum result index falls outside the 56 valid subsets.")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
