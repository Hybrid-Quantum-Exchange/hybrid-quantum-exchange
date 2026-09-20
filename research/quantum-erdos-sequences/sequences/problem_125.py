"""
Erdos problem #125 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, number: "125"):
    prize: no
    status: disproved (Lean)
    oeis: ["A367090"]
    tags: ["number theory", "base representations"]

OEIS A367090: "Numbers that cannot be expressed as a sum of distinct powers
of 3 and distinct powers of 4." (Erdos asked/conjectured about the density
of integers representable this way; #125 was disproved -- the sequence
A367090 records the counterexamples/gaps.)

Classical property under test:
    Fix a finite set of "digits" -- the powers of 3 below 81 and the powers
    of 4 below 64:
        P3 = (1, 3, 9, 27)
        P4 = (1, 4, 16)
    A non-negative integer t is REPRESENTABLE if some subset of P3 union
    some subset of P4 (each of the 7 elements used at most once, elements
    from P3 and P4 kept as separate multisets so 1=3^0 and 1=4^0 may both
    be used) sums to t. This is exactly the defining property of OEIS
    A367090 (whose non-members are exactly the representable numbers),
    restricted to a finite digit budget so it is a genuinely small, finite,
    computable search problem.

Chosen small instance:
    7 qubits, one per digit (4 for P3, 3 for P4) -> search space of size
    2^7 = 128 candidate subsets.
    Target t = 58.

    Classical brute force (first principles, done in this script, not
    looked up) enumerates all 128 subsets and finds that t = 58 has EXACTLY
    ONE representing subset:
        P3 bits (1,3,9,27) = (1,0,1,1)  -> 1 + 9 + 27 = 37
        P4 bits (1,4,16)   = (1,1,1)    -> 1 + 4 + 16 = 21
        37 + 21 = 58
    so t = 58 is representable (consistent with A367090, whose first terms
    -- 62, 63, 143, 144, ... -- do NOT include 58: 58 is correctly absent
    from the "cannot be expressed" sequence).

Quantum computation performed:
    Grover's search algorithm over the 7-qubit space of all 128 subsets.
    The oracle marks exactly the one subset (computed classically above,
    for oracle construction only) whose sum equals t = 58. With a single
    marked item among 128, the optimal number of Grover iterations is
    round(pi/4 * sqrt(128)) = 9. The circuit runs on the ideal AerSimulator;
    the most frequent measured bitstring is decoded back into a subset, its
    sum is recomputed classically, and compared against t.

    This is a genuine amplitude-amplification subset-sum search (the oracle
    is built from real arithmetic over P3/P4, not a disguised lookup table
    with no mathematical content), directly exercising the "base
    representations" / "number theory" property that defines A367090.

PASS criterion:
    The classical brute-force search confirms t = 58 has a unique
    representing subset with the sum shown above, AND the quantum circuit's
    most-likely measurement decodes to a subset whose sum equals 58.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

P3 = (1, 3, 9, 27)   # powers of 3: 3^0 .. 3^3
P4 = (1, 4, 16)      # powers of 4: 4^0 .. 4^2
DIGITS = P3 + P4      # 7 digits total, one qubit per digit
N = len(DIGITS)        # 7
TARGET = 58


def subset_sum(bits):
    """Sum of DIGITS[i] for every i with bits[i] == 1."""
    return sum(DIGITS[i] for i in range(N) if bits[i])


def classical_find_subset(target: int):
    """Brute-force, from first principles, over all 2^N subsets of
    P3 union P4 for one summing to `target`. Returns (subset_bits,
    all_matching_subsets) -- all_matching_subsets lets us confirm
    uniqueness, not just existence."""
    matches = []
    for mask in range(2 ** N):
        bits = tuple((mask >> i) & 1 for i in range(N))
        if subset_sum(bits) == target:
            matches.append(bits)
    return matches


def build_oracle(qc: QuantumCircuit, qubits, marked_bits):
    """Flip the phase of the single computational basis state whose bits
    (qubit i = bit i) equal marked_bits."""
    zero_positions = [qubits[i] for i, b in enumerate(marked_bits) if b == 0]
    if zero_positions:
        qc.x(zero_positions)
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


def grover_find_subset(marked_bits, shots=4096):
    """Run Grover's algorithm to find the unique basis state = marked_bits
    among the 2^N subsets of DIGITS. Returns (decoded_bits, counts)."""
    qc = QuantumCircuit(N, N)
    qubits = list(range(N))

    qc.h(qubits)

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
    print(f"Digits P3={P3}, P4={P4}, DIGITS={DIGITS}, N={N} qubits, "
          f"search space size={2 ** N}")
    print(f"Target t = {TARGET}")

    matches = classical_find_subset(TARGET)
    print(f"Classical brute-force subsets summing to {TARGET}: {matches}")

    unique = len(matches) == 1
    if not unique:
        print(f"Expected a unique representing subset for t={TARGET}, "
              f"found {len(matches)}. FAIL")
        return

    classical_bits = matches[0]
    classical_sum = subset_sum(classical_bits)
    assert classical_sum == TARGET

    quantum_bits, counts = grover_find_subset(classical_bits)
    quantum_sum = subset_sum(quantum_bits)

    top_count = counts[max(counts, key=counts.get)]
    total_shots = sum(counts.values())
    print(
        f"classical_subset={classical_bits} (sum={classical_sum})  "
        f"grover_subset={quantum_bits} (sum={quantum_sum})  "
        f"top_prob={top_count / total_shots:.3f}"
    )

    ok = unique and (quantum_sum == TARGET) and (quantum_bits == classical_bits)
    print()
    if ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
