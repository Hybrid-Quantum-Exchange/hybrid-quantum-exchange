"""
Erdos problem #701 (erdosproblems.com), tag: combinatorics / intersecting family.

Source-state note: as of the clone read at /home/user/manman4/erdosproblems
data/problems.yaml, problem #701 lists oeis: ["N/A"] -- there is no OEIS
sequence attached to this problem. Because the task requires deriving a
property from the problem's OEIS id(s) and tags, and there is no OEIS id
here, this script instead builds a genuine, finite, classically-checkable
instance of the underlying combinatorial notion named by the problem's own
tags ("combinatorics", "intersecting family"): the Erdos-Ko-Rado maximum
intersecting family problem. This is an honest best-effort substitute for
the missing OEIS anchor, not a copy of any OEIS value, and the limitation
(no OEIS id exists for #701) is recorded here explicitly.

Classical property tested
--------------------------
Let n = 4, k = 2. Let S be the list of all 2-element subsets ("edges") of
{0,1,2,3}; there are C(4,2) = 6 of them, enumerated below in a fixed order.
A "family" is any subset of S, encoded as a 6-bit string (bit i = 1 means
edge S[i] is included). A family is "intersecting" if every pair of edges
in it shares at least one element. The Erdos-Ko-Rado theorem says the
maximum size of an intersecting family of k-subsets of an n-set, for
n >= 2k, is C(n-1, k-1); here that is C(3,1) = 3.

The script first computes, purely classically by brute-force enumeration
over all 2^6 = 64 possible families, the exact set of 6-bit strings that
are (a) intersecting and (b) of the maximum possible size for an
intersecting family in this instance. This is the ground truth ("solution
set") against which the quantum result is checked -- it is derived from
first principles here, not looked up.

Quantum circuit
----------------
A Grover search circuit over 6 qubits (64 basis states) is built. The
oracle is a diagonal phase-flip oracle that marks exactly the classically
precomputed solution strings (implemented as multi-controlled Z gates, one
per solution bitstring, each preceded/followed by X gates on the 0-bits so
the multi-control fires only on that exact bitstring). The standard Grover
diffusion operator is applied for the optimal integer number of iterations
for this database size (64) and solution count (computed classically).
The circuit is run on the ideal AerSimulator (qasm-style sampling), and the
script checks that the most frequently measured bitstring is a member of
the classically-verified solution set, and that measured solution strings
receive the large majority of the amplified probability mass.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_max_intersecting_families(n: int, k: int):
    """Brute-force ground truth: all subsets of k-subsets of [n] that are
    pairwise-intersecting and of maximum size, plus the edge list itself."""
    edges = list(combinations(range(n), 2))
    m = len(edges)
    assert k == 2, "this helper only builds the k=2 edge case used below"

    best_size = 0
    intersecting_masks = []
    for mask in range(1 << m):
        chosen = [edges[i] for i in range(m) if (mask >> i) & 1]
        if not chosen:
            continue
        ok = True
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if not (set(chosen[i]) & set(chosen[j])):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            intersecting_masks.append((mask, len(chosen)))
            if len(chosen) > best_size:
                best_size = len(chosen)

    solutions = [mask for mask, size in intersecting_masks if size == best_size]
    return edges, m, best_size, sorted(solutions)


def build_oracle(num_qubits: int, solutions):
    """Diagonal phase oracle flipping the sign of exactly the given
    computational basis states (given as little-endian integers)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for sol in solutions:
        bits = [(sol >> i) & 1 for i in range(num_qubits)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def main():
    n, k = 4, 2
    edges, num_qubits, best_size, solutions = classical_max_intersecting_families(n, k)
    N = 1 << num_qubits
    M = len(solutions)

    print(f"Erdos problem #701: intersecting-family instance, n={n}, k={k}")
    print(f"Edges (2-subsets of [{n}]): {edges}")
    print(f"Search space size N = 2^{num_qubits} = {N}")
    print(f"Classical max intersecting family size (EKR predicts C(n-1,k-1) "
          f"= {math.comb(n - 1, k - 1)}): {best_size}")
    print(f"Number of maximum-size intersecting families (solutions): {M}")
    print(f"Solution bitstrings (little-endian ints): {solutions}")

    assert best_size == math.comb(n - 1, k - 1), (
        "classical brute force disagrees with the Erdos-Ko-Rado bound"
    )

    # Optimal number of Grover iterations for M solutions out of N.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(num_qubits, solutions)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit prints bit strings MSB-first with qubit 0 as the rightmost
    # character; convert back to the little-endian integer used above.
    def bitstr_to_int(bs: str) -> int:
        return int(bs[::-1], 2)

    solution_set = set(solutions)
    total_solution_prob = 0
    best_measured, best_measured_count = None, -1
    for bitstring, count in counts.items():
        val = bitstr_to_int(bitstring)
        if val in solution_set:
            total_solution_prob += count
        if count > best_measured_count:
            best_measured_count = count
            best_measured = val

    solution_fraction = total_solution_prob / shots
    top_is_solution = best_measured in solution_set

    print(f"Grover iterations used: {iterations}")
    print(f"Most frequent measured outcome: {best_measured} "
          f"(count {best_measured_count}/{shots}); is a solution: {top_is_solution}")
    print(f"Fraction of shots landing on a solution state: {solution_fraction:.4f}")

    # Success criteria: the amplified search concentrates on true solutions.
    passed = top_is_solution and solution_fraction > 0.5

    if passed:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
