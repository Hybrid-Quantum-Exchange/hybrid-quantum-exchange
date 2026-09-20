#!/usr/bin/env python3
"""
Erdos problem #139 -- quantum-testable sequence entry.

Source: /home/user/manman4/erdosproblems/data/problems.yaml, entry
  number: "139"
  comments: "Szemerédi's theorem"
  tags: ["additive combinatorics", "arithmetic progressions"]
  oeis: ["A003002", "A003003", "A003004", "A003005"]

Erdos problem #139 is Szemerédi's theorem on arithmetic progressions: any
subset of the integers with positive upper density contains arbitrarily long
arithmetic progressions. The associated OEIS family (A003002-A003005) is the
classical "3-free sequences" family: sequences over small alphabets (2 or 3
symbols) of maximal length containing no three equally-spaced equal terms.
The finite, computable kernel of this theorem that we test here is the
extremal-set analogue used to build that family and directly relevant to
density-based avoidance of 3-term APs:

    Classical property tested:
        For N = 6, let M = the maximum size of a subset S of {0, 1, ..., N-1}
        that contains NO 3-term arithmetic progression (no a < b < c in S
        with b - a == c - b). This is computed here by brute force over all
        2^N subsets (first principles, no OEIS lookup).

    Quantum task:
        Grover search over all 2^N = 64 bitstrings (each bit i =
        "element i is in S") for a subset that is simultaneously:
          (1) AP-3-free, and
          (2) of size exactly M (the classical maximum found above).
        The oracle is built directly from the classical brute-force check
        (marked states get phase -1), and the diffuser is the standard
        Grover diffusion operator. We run the optimal number of Grover
        iterations on the ideal AerSimulator and check that the most
        probable measured bitstring is indeed an AP-3-free subset of size M.

This is a genuine (if small) instance of the "find an extremal AP-free set"
search that underlies the classical study of Szemerédi-type theorems and the
A003002-A003005 OEIS family, verified against a first-principles classical
computation of the same instance.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Diagonal
from qiskit_aer import AerSimulator

N = 6  # number of integers {0, ..., N-1}; 2^N = 64 basis states


def has_ap3(subset):
    """True if subset (a sorted tuple of ints) contains a 3-term AP a<b<c, b-a==c-b."""
    s = sorted(subset)
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if s[j] - s[i] == s[k] - s[j]:
                    return True
    return False


def classical_max_ap3_free_size(n):
    """Brute force over all 2^n subsets of {0,...,n-1}; return the max size
    of a subset with no 3-term AP, and the list of bitmask indices attaining it.
    Bit i of the mask (0 = LSB) means element i is in the subset."""
    best_size = -1
    best_masks = []
    for mask in range(1 << n):
        subset = [i for i in range(n) if (mask >> i) & 1]
        if has_ap3(subset):
            continue
        size = len(subset)
        if size > best_size:
            best_size = size
            best_masks = [mask]
        elif size == best_size:
            best_masks.append(mask)
    return best_size, best_masks


def build_grover_circuit(n, marked_masks):
    """Build a Grover search circuit over n qubits marking the given basis
    states (by computational-basis index == bitmask), using the standard
    diagonal-phase-oracle + diffuser construction, run for the optimal
    number of iterations."""
    dim = 1 << n
    num_marked = len(marked_masks)

    # Diagonal phase oracle: -1 on marked basis states, +1 elsewhere.
    diag = [1.0] * dim
    for m in marked_masks:
        diag[m] = -1.0
    oracle_gate = Diagonal(diag)

    # Diffuser: standard Grover diffusion operator about the uniform state.
    diffuser = QuantumCircuit(n, name="diffuser")
    diffuser.h(range(n))
    diffuser.x(range(n))
    diffuser.h(n - 1)
    diffuser.mcx(list(range(n - 1)), n - 1)
    diffuser.h(n - 1)
    diffuser.x(range(n))
    diffuser.h(range(n))

    theta = math.asin(math.sqrt(num_marked / dim))
    iterations = max(1, round((math.pi / 4 / theta) - 0.5))

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.append(oracle_gate.to_instruction(), range(n))
        qc.append(diffuser.to_instruction(), range(n))
    qc.measure(range(n), range(n))
    return qc, iterations


def main():
    # 1. Classical ground truth, computed from first principles.
    best_size, best_masks = classical_max_ap3_free_size(N)
    print(f"N = {N}")
    print(f"Classical max AP-3-free subset size M = {best_size}")
    print(f"Number of subsets attaining M: {len(best_masks)}")
    example = [i for i in range(N) if (best_masks[0] >> i) & 1]
    print(f"Example extremal AP-3-free subset: {example}")

    # 2. Build and run the Grover circuit targeting exactly those subsets.
    qc, iterations = build_grover_circuit(N, best_masks)
    print(f"Grover iterations used: {iterations}")

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Qiskit bit ordering: rightmost char of the bitstring is qubit 0.
    def bitstring_to_mask(bs):
        return int(bs[::-1], 2)

    most_common_bs = max(counts, key=counts.get)
    most_common_mask = bitstring_to_mask(most_common_bs)
    most_common_subset = tuple(i for i in range(N) if (most_common_mask >> i) & 1)

    total_shots = sum(counts.values())
    marked_shots = sum(c for bs, c in counts.items() if bitstring_to_mask(bs) in best_masks)
    print(f"Most frequent measured mask: {most_common_mask} -> subset {most_common_subset}")
    print(f"Fraction of shots landing on a marked (extremal, AP-3-free) state: "
          f"{marked_shots}/{total_shots} = {marked_shots / total_shots:.3f}")

    # 3. Verify: the most probable outcome must be AP-3-free and of size M.
    quantum_ok = (
        most_common_mask in best_masks
        and not has_ap3(most_common_subset)
        and len(most_common_subset) == best_size
    )
    # Also require that amplification actually concentrated probability
    # (majority of shots land on a marked state), not just a lucky top pick.
    amplification_ok = marked_shots / total_shots > 0.5

    verified = quantum_ok and amplification_ok

    print()
    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
