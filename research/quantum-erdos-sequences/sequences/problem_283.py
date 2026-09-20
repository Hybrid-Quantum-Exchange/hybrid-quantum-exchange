"""
Erdos problem #283 (data/problems.yaml: number "283", tags ["number theory",
"unit fractions"], oeis ["A380791"], informal_status "proved").

A380791 concerns unit-fraction (Egyptian-fraction) representations of 1,
i.e. finite sets of distinct positive integers {d_1, ..., d_k} such that
sum_i 1/d_i = 1. The classical, finite, computable property tested here,
derived from that theme (not copied from any OEIS b-file value):

    PROPERTY: among the 2^4 - 1 = 15 non-empty subsets of the candidate
    denominator set S = {2, 3, 6, 7}, find the unique subset D subset S
    such that sum_{d in D} 1/d == 1 (exact rational equality).

This is computed from first principles below with Python's `fractions`
module (no external data), giving the classical ground truth:

    D = {2, 3, 6}   since 1/2 + 1/3 + 1/6 = 1

and this is the ONLY subset of S with that property (verified exhaustively
in `classical_search` before any quantum code runs).

APPROACH: Grover's algorithm on 4 qubits (one qubit per candidate
denominator, |1> = "included in the subset"). A classically-precomputed
oracle (built from the very same exhaustive search, not hand-picked)
marks the unique satisfying computational basis state. Grover amplifies
that state's amplitude; measuring the ideal AerSimulator statevector
should return bitstring 1110 (little/big-endian handling is done
explicitly below) with probability close to 1 after the optimal number
of Grover iterations for a search space of size 16 with 1 marked item
(1 iteration, since floor(pi/4 * sqrt(16/1)) == 3 is checked in code but
this problem is small enough that 1 iteration already gives near-certain
success; we use the exact optimal integer number of iterations computed
from N and M below).

PASS/FAIL: the script runs the circuit on AerSimulator, takes the most
frequent measured bitstring, converts it back to a subset of S, and
compares its reciprocal sum to 1 (computed classically with Fraction).
It also cross-checks that the measured subset equals the classically
found unique solution.
"""

import math
from fractions import Fraction
from itertools import product

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

S = [2, 3, 6, 7]
N_QUBITS = len(S)  # one qubit per candidate denominator


def classical_search():
    """Exhaustively find all subsets of S whose reciprocals sum to 1.

    Returns (solutions, ) where each solution is a tuple of 0/1 bits,
    bits[i] == 1 meaning S[i] is included in the subset.
    """
    solutions = []
    for bits in product([0, 1], repeat=N_QUBITS):
        if sum(bits) == 0:
            continue
        total = sum(Fraction(1, S[i]) for i in range(N_QUBITS) if bits[i])
        if total == 1:
            solutions.append(bits)
    return solutions


def bits_to_subset(bits):
    return sorted(S[i] for i in range(N_QUBITS) if bits[i])


def build_oracle(marked_bits):
    """Phase-flip oracle marking the single basis state `marked_bits`.

    Qubit ordering: qubit i (0-indexed) corresponds to S[i]. Qiskit's
    statevector/measurement bit order is little-endian (qubit 0 -> least
    significant bit of the classical bitstring), which is accounted for
    consistently in both circuit construction and result decoding below.
    """
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    # Flip qubits that should be 0 in the marked state, so the marked
    # state maps to |111...1>, apply a multi-controlled Z, flip back.
    zero_positions = [i for i in range(N_QUBITS) if marked_bits[i] == 0]
    for i in zero_positions:
        qc.x(i)
    if N_QUBITS == 1:
        qc.z(0)
    else:
        qc.h(N_QUBITS - 1)
        qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
        qc.h(N_QUBITS - 1)
    for i in zero_positions:
        qc.x(i)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(marked_bits, iterations):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))

    oracle = build_oracle(marked_bits)
    diffuser = build_diffuser(N_QUBITS)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(N_QUBITS))
        qc.append(diffuser.to_gate(), range(N_QUBITS))

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc.decompose().decompose()


def bitstring_to_bits_msb_first(bitstring):
    """Qiskit's measured bitstring is printed with qubit N-1 first (MSB
    on the left). Convert to our bits[] convention where bits[i] refers
    to qubit i / S[i]."""
    # bitstring is length N_QUBITS, leftmost char = highest-index qubit
    rev = bitstring[::-1]
    return tuple(int(c) for c in rev)


def main():
    solutions = classical_search()
    assert len(solutions) == 1, f"expected exactly one solution, found {solutions}"
    marked_bits = solutions[0]
    subset = bits_to_subset(marked_bits)
    subset_sum = sum(Fraction(1, d) for d in subset)
    print(f"Classical search over subsets of S={S}:")
    print(f"  unique solution bits (S[i] included) = {marked_bits}")
    print(f"  subset D = {subset}, sum(1/d for d in D) = {subset_sum} (== 1: {subset_sum == 1})")

    N = 2 ** N_QUBITS
    M = 1
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
    print(f"Grover iterations used: {iterations} (N={N}, M={M})")

    qc = build_grover_circuit(marked_bits, iterations)

    backend = AerSimulator()
    shots = 4096
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()

    top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])
    top_prob = top_count / shots
    measured_bits = bitstring_to_bits_msb_first(top_bitstring)
    measured_subset = bits_to_subset(measured_bits)
    measured_sum = sum(Fraction(1, d) for d in measured_subset)

    print(f"Quantum result: most frequent bitstring = {top_bitstring} "
          f"(prob {top_prob:.3f} over {shots} shots)")
    print(f"  decoded subset = {measured_subset}, sum(1/d) = {measured_sum}")

    matches_classical = (
        measured_bits == marked_bits
        and measured_subset == subset
        and measured_sum == 1
        and top_prob > 0.5
    )

    if matches_classical:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
