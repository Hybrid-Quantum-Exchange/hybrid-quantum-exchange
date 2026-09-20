"""
Erdos problem #781 -- quantum-testable sequence entry.

Source metadata (data/problems.yaml in the manman4/erdosproblems clone, entry
"number: '781'", verified 2026-09-19):
    prize: no
    informal_status: disproved (last_update 2025-08-31)
    oeis: ["possible"]
    tags: ["additive combinatorics"]

LIMITATION, stated honestly up front: the "oeis" field for problem 781 in the
source data is the literal string "possible", not a real OEIS sequence id.
There is no A-number to anchor a sequence-membership test to, and no problem
detail file exists in the clone (checked: no docs/problems/*781* file). So
this script does NOT test any actual OEIS sequence -- it would be fabrication
to pretend otherwise. Per the task's own fallback instruction ("if no OEIS
id ... no genuine quantum circuit can be constructed for this problem's
sequence, write the script anyway with your best honest attempt"), this
script instead builds a genuine, real Grover-search quantum circuit for a
small, finite, computable property that sits squarely inside the problem's
own tag, "additive combinatorics": finding a SUM-FREE SUBSET of {1,2,3,4}.

Classical property being tested
--------------------------------
A subset S of {1,2,3,4} is "sum-free" if there is no solution a+b=c with
a, b, c all in S (a and b need not be distinct, so e.g. {1,2} is NOT
sum-free because 1+1=2). The empty set and every singleton are trivially
sum-free, so to give Grover's amplitude amplification something meaningful
to do we search for NON-TRIVIAL sum-free subsets: sum-free AND |S| >= 2.
We classically enumerate all 2^4 = 16 subsets from first principles,
determine exactly which are non-trivially sum-free, and use that ground
truth in two ways:
  1. to build the Grover oracle (marking exactly those subsets), and
  2. as the answer we check the quantum result against.

Circuit
-------
4 qubits, one per element of {1,2,3,4} (bit i = 1 means element i+1 is in S).
Grover's algorithm: uniform superposition -> oracle (multi-controlled phase
flip on each sum-free basis state, built directly from the classically
derived list) -> diffusion operator, repeated the optimal number of times
for a 4-qubit search space and a known number of marked items. Measurement
on AerSimulator should overwhelmingly land on a sum-free subset.

PASS/FAIL: the script passes if the most frequently measured 4-bit string,
across many shots, decodes to a subset that the from-scratch classical
brute force also certifies as sum-free.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


ELEMENTS = [1, 2, 3, 4]
N_QUBITS = len(ELEMENTS)


def is_sum_free(subset):
    """True iff no a,b,c in subset (a,b need not be distinct) satisfy a+b=c."""
    s = set(subset)
    for a in s:
        for b in s:
            if (a + b) in s:
                return False
    return True


def is_nontrivial_sum_free(subset):
    """Sum-free AND has at least 2 elements (excludes the trivially sum-free
    empty set and singletons, which would otherwise make "sum-free" true for
    roughly half of all subsets and leave nothing for Grover to amplify)."""
    return len(subset) >= 2 and is_sum_free(subset)


def classical_sum_free_subsets():
    """Brute-force, from first principles, every sum-free subset of ELEMENTS.

    Returns a sorted list of 4-bit strings (bit i = element ELEMENTS[i]
    present), little-endian in qubit index (qubit 0 = element 1, etc.).
    """
    marked = []
    for mask in range(2 ** N_QUBITS):
        subset = [ELEMENTS[i] for i in range(N_QUBITS) if (mask >> i) & 1]
        if is_nontrivial_sum_free(subset):
            # bitstring as Qiskit prints it: qubit (N-1) ... qubit 0
            bits = "".join(str((mask >> i) & 1) for i in reversed(range(N_QUBITS)))
            marked.append(bits)
    return sorted(marked)


def build_oracle(marked_bitstrings, n_qubits):
    """Phase-flip oracle marking exactly the given basis states."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in marked_bitstrings:
        # bits[0] is qubit n-1 ... bits[-1] is qubit 0 (Qiskit convention)
        zero_positions = [n_qubits - 1 - i for i, c in enumerate(bits) if c == "0"]
        for q in zero_positions:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run():
    marked = classical_sum_free_subsets()
    n_marked = len(marked)
    n_total = 2 ** N_QUBITS

    print(f"Classical brute force over all {n_total} subsets of {ELEMENTS}:")
    for bits in marked:
        subset = [ELEMENTS[i] for i in range(N_QUBITS) if bits[N_QUBITS - 1 - i] == "1"]
        print(f"  non-trivial sum-free subset {subset!r}  (bitstring {bits})")
    print(f"Total non-trivial sum-free subsets (classical ground truth): {n_marked}")

    if n_marked == 0 or n_marked == n_total:
        raise RuntimeError("degenerate search space; cannot run Grover meaningfully")

    # Optimal number of Grover iterations for n_marked out of n_total items.
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked, N_QUBITS)
    diffuser = build_diffuser(N_QUBITS)

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(N_QUBITS))
        qc.append(diffuser.to_gate(), range(N_QUBITS))
    qc.measure(range(N_QUBITS), range(N_QUBITS))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    most_common_bits, most_common_count = max(counts.items(), key=lambda kv: kv[1])
    prob_marked = sum(c for b, c in counts.items() if b in marked) / shots

    print(f"\nGrover iterations used: {iterations}")
    print(f"Most frequent measured bitstring: {most_common_bits} "
          f"({most_common_count}/{shots} shots)")
    print(f"Fraction of shots landing on a classically-verified non-trivial "
          f"sum-free subset: {prob_marked:.3f}")

    quantum_found_sum_free = most_common_bits in marked
    amplification_worked = prob_marked > (n_marked / n_total) * 1.5

    ok = quantum_found_sum_free and amplification_worked
    print("\nPASS" if ok else "\nFAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
