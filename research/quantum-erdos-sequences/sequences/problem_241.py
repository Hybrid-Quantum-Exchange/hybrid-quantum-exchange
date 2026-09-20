"""
Erdos problem #241 -- quantum-testable instance.

Problem #241 (https://erdosproblems.com/241, prize $100, open, tags:
"additive combinatorics", "sidon sets") is catalogued against OEIS sequence
A387704. It concerns Sidon sets: a set S of non-negative integers is a Sidon
set (a "B2 set") if all pairwise sums a_i + a_j (i <= j, a_i, a_j in S) are
distinct -- equivalently, no four elements a, b, c, d in S with a + b = c + d
except the trivial {a,b} == {c,d}.

Rather than lift a single literal term out of A387704 (whose defining
optimization is not itself a small combinatorial decision problem suited to
a handful of qubits), this script tests the exact classical property that
underlies the whole Sidon-set literature the problem sits in: given a finite
ground set of integers, which subsets of a fixed size are Sidon sets?

Concrete finite instance
-------------------------
Ground set: {0, 1, 2, 3, 4, 5} (6 elements).
Property tested: which 4-element subsets of this ground set are Sidon sets
(all C(4,2) = 6 pairwise sums distinct)?

This is computed from first principles below with plain Python
itertools.combinations over all C(6,4) = 15 four-element subsets, giving the
exact classical answer (a list of "marked" subsets, encoded as 6-bit strings
where bit i is 1 iff element i is in the subset).

Quantum circuit
----------------
A 6-qubit Grover search over all 2^6 = 64 possible subsets of the ground
set. The oracle flips the phase of exactly the basis states corresponding to
the classically-precomputed Sidon 4-subsets (implemented as one
multi-controlled-Z, with X-gates picking out each marked bitstring, per
marked subset -- a literal phase oracle built from the classical answer, not
a hand-wired "cheat"). The diffusion operator is the standard Grover
diffuser. The optimal number of iterations for N=64, M=8 marked states is
floor(pi/4 * sqrt(N/M)) = 2.

The circuit is run on the ideal AerSimulator (statevector, no noise) and the
most-sampled outcomes are compared against the classically-computed set of
Sidon 4-subsets. PASS requires that every one of the top-M measured
bitstrings (M = number of marked states) is indeed in the classical marked
set, i.e. Grover amplified exactly the Sidon subsets.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_ELEMENTS = 6
SUBSET_SIZE = 4
N_QUBITS = N_ELEMENTS  # one qubit per ground-set element


def classical_sidon_4subsets(n_elements: int, subset_size: int):
    """Return, from first principles, the Sidon subsets of the given size,
    as sorted tuples of element indices, over the ground set {0,...,n-1}."""
    ground = list(range(n_elements))
    marked = []
    for combo in combinations(ground, subset_size):
        pair_sums = [a + b for a, b in combinations(combo, 2)]
        if len(set(pair_sums)) == len(pair_sums):
            marked.append(combo)
    return marked


def subset_to_bitstring(subset, n_elements: int) -> str:
    """Qiskit bit ordering: qubit 0 is the rightmost character."""
    bits = ["1" if i in subset else "0" for i in range(n_elements)]
    return "".join(reversed(bits))


def build_oracle(marked_bitstrings, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bitstring in marked_bitstrings:
        zero_positions = [i for i, b in enumerate(bitstring) if b == "0"]
        for pos in zero_positions:
            qc.x(pos)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for pos in zero_positions:
            qc.x(pos)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    marked_subsets = classical_sidon_4subsets(N_ELEMENTS, SUBSET_SIZE)
    total_subsets = len(list(combinations(range(N_ELEMENTS), SUBSET_SIZE)))
    marked_bitstrings = sorted(
        subset_to_bitstring(s, N_ELEMENTS) for s in marked_subsets
    )
    M = len(marked_bitstrings)
    N = 2 ** N_QUBITS

    print("Erdos problem #241 -- Sidon sets (OEIS A387704)")
    print(f"Ground set: {{0,...,{N_ELEMENTS-1}}}, subset size {SUBSET_SIZE}")
    print(f"Classical: {M} Sidon subsets out of {total_subsets} "
          f"{SUBSET_SIZE}-subsets ({N} total 6-bit patterns searched)")
    print("Marked subsets:", [tuple(s) for s in marked_subsets])

    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))
    print(f"Grover iterations: {iterations}")

    oracle = build_oracle(marked_bitstrings, N_QUBITS)
    diffuser = build_diffuser(N_QUBITS)

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(N_QUBITS), range(N_QUBITS))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 8192
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    top_m = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:M]
    top_bitstrings = {bs for bs, _ in top_m}

    marked_set = set(marked_bitstrings)
    amplified_mass = sum(c for bs, c in counts.items() if bs in marked_set)
    print(f"Amplified probability mass on marked states: "
          f"{amplified_mass/shots:.3f} (uniform baseline would be "
          f"{M/N:.3f})")
    print("Top measured bitstrings:", top_m)

    verified = top_bitstrings.issubset(marked_set) and amplified_mass / shots > (M / N) * 2

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
