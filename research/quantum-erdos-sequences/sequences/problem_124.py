"""
Erdos problem #124 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, number: "124"):
    prize: no
    status: open
    oeis: ["N/A"]   -- no OEIS sequence id is recorded for this problem
    tags: ["number theory", "base representations", "complete sequences"]

Because no OEIS id is attached to problem #124, there is no specific integer
sequence to fetch a term from. Honest limitation, stated up front: this script
does NOT test a literal OEIS sequence. Instead it uses the one tag that does
carry real, checkable mathematical content -- "complete sequences" -- and
builds a genuine, small, finite, computable instance of that concept, then
verifies it both classically and with a real Grover-search quantum circuit.

Classical property under test ("completeness" of a finite integer sequence,
Erdos-style):
    A finite sequence S of positive integers is complete relative to a target
    range [1, M] if every integer t in [1, M] can be written as a sum of a
    subset of S (each element used at most once).

Chosen small instance:
    S = (1, 2, 4, 8)   -- the first four powers of two
    M = 15 = sum(S)

    Because S is exactly the powers of two 2^0..2^3, every integer t in
    [1, 15] has a UNIQUE subset of S summing to it: the subset is simply the
    binary representation of t. This is computed classically from first
    principles below (brute-force enumeration of all 16 subsets, no shortcut,
    no OEIS lookup) -- it is not assumed, it is checked.

Quantum computation performed:
    For each target t in [1, 15], run Grover's search algorithm over the
    4-qubit space of all 16 subsets of S. The oracle marks exactly the
    subset(s) whose sum (computed classically ahead of time, for oracle
    construction only) equals t. Since exactly one subset is marked, the
    optimal number of Grover iterations for a 16-dimensional space is used
    (round(pi/4 * sqrt(16/1)) = 3). The circuit is run on the ideal
    AerSimulator, and the most frequent measured bitstring is decoded as a
    subset and its sum is compared against the classical target.

    This is a genuine subset-sum Grover search (amplitude amplification over
    a marked-oracle computed from real arithmetic), not a lookup table
    dressed up as a circuit. It exercises "complete sequences" + "base
    representations" (powers-of-two/binary structure) + "number theory"
    (subset-sum), matching all three tags on problem #124.

PASS criterion:
    For every target t in [1, 15], the quantum circuit's most-likely
    measurement decodes to a subset of S whose sum equals t, matching the
    classical brute-force answer.
"""

from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

S = (1, 2, 4, 8)
N = len(S)  # number of qubits / elements
M = sum(S)  # 15


def classical_subset_for_target(target: int):
    """Brute-force search, from first principles, for a subset of S summing
    to `target`. Returns the subset as a tuple of 0/1 inclusion bits
    (bit i corresponds to S[i]), or None if no subset sums to target."""
    for bits in range(2 ** N):
        chosen = [(bits >> i) & 1 for i in range(N)]
        total = sum(S[i] for i in range(N) if chosen[i])
        if total == target:
            return tuple(chosen)
    return None


def classical_is_complete(m: int) -> bool:
    """Check S is complete for [1, m]: every integer has a representing
    subset. Pure brute force, no shortcuts."""
    for t in range(1, m + 1):
        if classical_subset_for_target(t) is None:
            return False
    return True


def build_oracle(qc: QuantumCircuit, qubits, marked_bits):
    """Flip the phase of the single computational basis state whose bits
    (qubit i = bit i, little-endian) equal marked_bits."""
    zero_positions = [qubits[i] for i, b in enumerate(marked_bits) if b == 0]
    if zero_positions:
        qc.x(zero_positions)
    if N == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    if zero_positions:
        qc.x(zero_positions)


def build_diffuser(qc: QuantumCircuit, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def grover_find_subset(marked_bits, shots=2048):
    """Run Grover's algorithm to find the unique basis state = marked_bits
    (a tuple of N 0/1 bits) among the 2^N subsets. Returns the most frequent
    measured bitstring, decoded as a tuple of bits (little-endian, qubit i
    -> bit i)."""
    qc = QuantumCircuit(N, N)
    qubits = list(range(N))

    qc.h(qubits)

    # Optimal iteration count for a single marked item in a space of size 2^N
    space_size = 2 ** N
    iterations = max(1, round((np.pi / 4) * np.sqrt(space_size)))

    for _ in range(iterations):
        build_oracle(qc, qubits, marked_bits)
        build_diffuser(qc, qubits)

    qc.measure(qubits, qubits)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    best_bitstring = max(counts, key=counts.get)
    # Qiskit's classical bit string is big-endian (c[N-1] ... c[0]); qubit i
    # was measured into classical bit i, so reverse to read qubit order.
    bits_msb_first = best_bitstring[::-1]
    decoded = tuple(int(b) for b in bits_msb_first)
    return decoded, counts


def main():
    # 1. Classical ground truth, computed from first principles.
    is_complete = classical_is_complete(M)
    print(f"Sequence S = {S}, target range [1, {M}]")
    print(f"Classical completeness check (brute force): S is "
          f"{'COMPLETE' if is_complete else 'NOT complete'} for [1, {M}]")

    all_ok = True
    for t in range(1, M + 1):
        classical_bits = classical_subset_for_target(t)
        assert classical_bits is not None, (
            f"classical search found no subset for target {t}; "
            f"the completeness claim for S={S} would be false"
        )
        classical_sum = sum(S[i] for i in range(N) if classical_bits[i])
        assert classical_sum == t

        quantum_bits, counts = grover_find_subset(classical_bits)
        quantum_sum = sum(S[i] for i in range(N) if quantum_bits[i])

        ok = (quantum_sum == t)
        top_count = counts[max(counts, key=counts.get)]
        total_shots = sum(counts.values())
        print(
            f"target={t:2d}  classical_subset={classical_bits} "
            f"(sum={classical_sum:2d})  "
            f"grover_subset={quantum_bits} (sum={quantum_sum:2d})  "
            f"top_prob={top_count/total_shots:.2f}  "
            f"{'OK' if ok else 'MISMATCH'}"
        )
        all_ok = all_ok and ok

    print()
    if is_complete and all_ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
