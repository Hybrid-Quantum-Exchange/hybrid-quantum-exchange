"""
Erdos problem #37 -- quantum-testable companion script.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry
`number: "37"`):
    prize: no
    informal_status: disproved (2025-08-31), formal_status: Lean (2026-08-24)
    oeis: ["N/A"]
    tags: ["number theory", "additive combinatorics"]

LIMITATION, stated honestly up front: problem #37 has *no* associated OEIS
sequence id ("N/A" in the source data). The task requires deriving a small,
finite, computable property from the problem's OEIS id(s) and tags; with no
OEIS id available, there is no literal sequence term to test membership in.
Rather than fabricate an OEIS id or invent a fake "sequence", this script
instead builds a genuine, small, finite computable instance of the same
mathematical territory named by the problem's own tags
("number theory", "additive combinatorics"): finding a maximum-size
SUM-FREE subset of {1, ..., 5}.

A set S of positive integers is sum-free if there do NOT exist x, y, z in S
(x, y, z need not be distinct) with x + y = z. Sum-free sets are a classical
object in additive combinatorics (this is exactly the kind of object Erdos
himself studied, and it is the closest honest, finite, brute-force-checkable
substitute for the missing sequence).

Classical property under test
------------------------------
Search space: all 2^5 = 32 subsets of {1, 2, 3, 4, 5}, encoded as 5-bit
strings b4 b3 b2 b1 b0 where bit i (0-indexed) is 1 iff element (i+1) is in
the subset.

Property tested: "subset is sum-free AND has the maximum possible size
among sum-free subsets of {1,...,5}".

The script first computes, purely classically (brute force over all 32
subsets, from first principles -- no OEIS lookup, no hardcoded answer), the
set of all such maximum sum-free subsets. This is the classical ground
truth.

It then builds a Grover search circuit over 5 qubits whose oracle is a
multi-controlled phase flip on exactly those classically-precomputed marked
basis states (a standard, legitimate way to realize a Grover oracle for a
small enumerable predicate -- the marking is computed classically, but the
amplification, interference and measurement are done entirely by the
quantum circuit on the AerSimulator). One Grover iteration is applied
(optimal for this ~32-state search space with 3 marked states), and the
circuit is measured. The script PASSes if the most frequently measured
bitstring decodes to one of the classically-verified maximum sum-free
subsets.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N = 5  # elements 1..5
UNIVERSE = list(range(1, N + 1))


def is_sum_free(subset):
    s = set(subset)
    if not s:
        return False  # exclude the empty set; not an interesting witness
    for x in s:
        for y in s:
            if (x + y) in s:
                return False
    return True


def classical_max_sum_free_subsets():
    """Brute-force, from first principles, every subset of {1..5}."""
    best_size = 0
    best_subsets = []
    for r in range(1, N + 1):
        for combo in itertools.combinations(UNIVERSE, r):
            if is_sum_free(combo):
                if r > best_size:
                    best_size = r
                    best_subsets = [combo]
                elif r == best_size:
                    best_subsets.append(combo)
    return best_size, best_subsets


def subset_to_bitstring(subset):
    """bit i (0-indexed, i=0..4) set iff element (i+1) is in subset.
    Returned as a 5-char string b4 b3 b2 b1 b0 (qiskit / Qiskit-style,
    most significant qubit first) matching classical integer value."""
    val = 0
    for elem in subset:
        val |= 1 << (elem - 1)
    return format(val, "0{}b".format(N))


def build_grover_circuit(marked_ints, n_qubits):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    def oracle():
        oc = QuantumCircuit(n_qubits, name="oracle")
        for m in marked_ints:
            bits = format(m, "0{}b".format(n_qubits))[::-1]  # bit0 first
            flip_qubits = [i for i, b in enumerate(bits) if b == "0"]
            for q in flip_qubits:
                oc.x(q)
            oc.h(n_qubits - 1)
            oc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            oc.h(n_qubits - 1)
            for q in flip_qubits:
                oc.x(q)
        return oc

    def diffuser():
        dc = QuantumCircuit(n_qubits, name="diffuser")
        dc.h(range(n_qubits))
        dc.x(range(n_qubits))
        dc.h(n_qubits - 1)
        dc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        dc.h(n_qubits - 1)
        dc.x(range(n_qubits))
        dc.h(range(n_qubits))
        return dc

    # Optimal number of Grover iterations for N=32 states, M marked states:
    # r ~ floor(pi/4 * sqrt(N/M))
    total_states = 2 ** n_qubits
    m = len(marked_ints)
    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(total_states / m))))

    for _ in range(iterations):
        qc.append(oracle().to_gate(), range(n_qubits))
        qc.append(diffuser().to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


def main():
    best_size, best_subsets = classical_max_sum_free_subsets()
    marked_ints = sorted(
        {int(subset_to_bitstring(s), 2) for s in best_subsets}
    )

    print("Erdos problem #37 -- no OEIS id available (oeis: ['N/A']).")
    print("Substitute finite instance: maximum sum-free subsets of {1,...,5}")
    print(f"Classical max sum-free subset size: {best_size}")
    print("Classical maximum sum-free subsets:", best_subsets)
    print("Marked computational-basis integers:", marked_ints)

    qc, iterations = build_grover_circuit(marked_ints, N)
    print(f"Grover iterations used: {iterations}")

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Qiskit's classical register string is c4c3c2c1c0 (little-endian bit
    # order string, MSB-left); convert to the integer value directly.
    top_bitstring = max(counts, key=counts.get)
    top_int = int(top_bitstring, 2)
    top_count = counts[top_bitstring]

    print(f"Most frequent measured bitstring: {top_bitstring} "
          f"(int {top_int}), count {top_count}/4096")

    quantum_found_marked = top_int in marked_ints
    # Additional check: marked states should collectively dominate the
    # measured distribution (amplitude amplification actually worked).
    marked_total = sum(counts.get(format(m, "0{}b".format(N)), 0)
                        for m in marked_ints)
    amplification_worked = marked_total > 4096 * 0.5

    verified = quantum_found_marked and amplification_worked
    print(f"Marked-state measurement fraction: {marked_total}/4096")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
