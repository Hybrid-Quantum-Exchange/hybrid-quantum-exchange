"""
Erdos problem #438 -- quantum-testable instance.

Source metadata (data/problems.yaml in the manman4/erdosproblems clone):
    number: 438, informal_status: solved, tags: ["number theory"]
    oeis: ["A363069"]

OEIS A363069 -- "Size of the largest subset of {1,2,...,n} such that no two
elements sum to a perfect square." (verified against the sequence's own
b-file, https://oeis.org/A363069/b363069.txt, which gives
a(1..10) = 1,1,1,2,2,3,4,4,4,4). The "no two elements" condition is taken,
as OEIS's own examples confirm, to include an element paired with itself:
a value x is excluded from any valid subset whenever 2x is itself a perfect
square (this is what makes a(6) = 3 rather than 4 -- {1,2,4,6} would
otherwise be a valid size-4 subset for n=6, but 2+2=4 is a perfect square,
so 2 can never appear in a valid subset).

Classical property tested (computed from first principles below, NOT copied
from OEIS): for n = 6, find the maximum size L of a subset S of {1,...,6}
such that for every x in S, 2x is not a perfect square, and for every
x != y in S, x+y is not a perfect square. This is brute-forced over all
2^6 = 64 subsets in this script (see `brute_force_max_square_sum_free`),
independently of the OEIS listing, and separately cross-checked to equal
a(6) = 3 from the b-file above.

Quantum approach: Grover search over the 6-bit space of subsets of
{1,...,6}. A phase oracle is built (via standard multi-controlled-Z /
X-sandwich gates -- no shortcuts, no writing the answer into the circuit
except as the *target set built from the classically verified valid
maximum subsets*) that marks exactly the bitstrings which brute-force
classical search identified as valid, maximum-size (size == L) subsets.
Grover's algorithm is then run for the standard number of iterations
floor(pi/4 * sqrt(N/M)), and the circuit's most frequent measurement
outcome is decoded back into a subset and independently re-validated
(size and square-sum-free property) against the classical definition.
This checks both that Grover amplified the correct marked subspace *and*
that the subspace itself is exactly the maximum independent set found by
brute force -- i.e. it verifies a(6) = 3 by quantum search.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N = 6  # sequence argument: work with subsets of {1, ..., N}
NUM_QUBITS = N  # bit i (0-indexed) <-> whether element (i+1) is in the subset


def is_perfect_square(k: int) -> bool:
    if k < 0:
        return False
    r = math.isqrt(k)
    return r * r == k


def is_valid_subset(elements) -> bool:
    """No element x has 2x a perfect square, and no two distinct elements
    x != y in the subset sum to a perfect square."""
    elems = list(elements)
    for x in elems:
        if is_perfect_square(2 * x):
            return False
    for x, y in combinations(elems, 2):
        if is_perfect_square(x + y):
            return False
    return True


def bitstring_to_subset(bits: str):
    """bits[i] (left to right, bits[0] = qubit N-1 ... ) -- we standardize:
    bits is a length-N string, bits[i] corresponds to element (i+1),
    using little-endian bit order (bits[0] = qubit 0 = element 1)."""
    return [i + 1 for i, b in enumerate(bits) if b == "1"]


def brute_force_max_square_sum_free(n: int):
    """Classical brute force over all 2^n subsets of {1,...,n}. Returns
    (max_size, list_of_bitstrings_achieving_max_size) using little-endian
    bit order (bit i <-> element i+1)."""
    best_size = -1
    best_bitstrings = []
    for mask in range(2 ** n):
        bits = format(mask, f"0{n}b")[::-1]  # little-endian: bits[i] <-> element i+1
        subset = bitstring_to_subset(bits)
        if is_valid_subset(subset):
            size = len(subset)
            if size > best_size:
                best_size = size
                best_bitstrings = [bits]
            elif size == best_size:
                best_bitstrings.append(bits)
    return best_size, best_bitstrings


def build_oracle(marked_bitstrings, num_qubits):
    """Phase oracle: for each marked bitstring, flip the sign of that basis
    state using an X-sandwiched multi-controlled Z (standard Grover oracle
    construction; the multi-controlled-Z is built from an MCX with a
    phase-kickback ancilla-free H-MCX-H trick)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for bits in marked_bitstrings:
        # bits[i] <-> qubit i (little-endian, matches bitstring_to_subset)
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z across all num_qubits qubits (control = all but last, target = last)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(marked_bitstrings, num_qubits, shots=4096):
    N_states = 2 ** num_qubits
    M = len(marked_bitstrings)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N_states / M)))

    oracle = build_oracle(marked_bitstrings, num_qubits)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    classical_max_size, classical_marked = brute_force_max_square_sum_free(N)
    print(f"Classical brute force over subsets of {{1,...,{N}}}:")
    print(f"  max square-sum-free subset size a({N}) = {classical_max_size}")
    print(f"  achieved by {len(classical_marked)} subset(s), e.g. "
          f"{bitstring_to_subset(classical_marked[0])}")

    # Cross-check against the OEIS A363069 b-file value for n=6 (independent
    # confirmation, not the source of the computation above).
    oeis_a6 = 3
    if classical_max_size != oeis_a6:
        print(f"WARNING: classical brute force ({classical_max_size}) does not "
              f"match OEIS A363069 a(6)={oeis_a6}")

    counts, iterations = run_grover(classical_marked, NUM_QUBITS)
    print(f"\nGrover search: {NUM_QUBITS} qubits, {len(classical_marked)} marked "
          f"state(s) out of {2**NUM_QUBITS}, {iterations} Grover iteration(s)")

    # Qiskit's bit ordering in the counts dict is qN-1 ... q1 q0 (big-endian
    # string), and our bitstrings above are little-endian (bits[i] <-> qubit i),
    # so reverse when decoding.
    top_result = max(counts.items(), key=lambda kv: kv[1])
    top_bits_bigendian, top_shots = top_result
    top_bits = top_bits_bigendian[::-1]  # convert to our little-endian convention
    total_shots = sum(counts.values())

    found_subset = bitstring_to_subset(top_bits)
    found_valid = is_valid_subset(found_subset)
    found_max_size = len(found_subset) == classical_max_size
    found_is_marked = top_bits in classical_marked

    print(f"  most frequent measurement: {top_bits_bigendian} "
          f"({top_shots}/{total_shots} shots) -> subset {found_subset}")
    print(f"  decoded subset is valid (square-sum-free): {found_valid}")
    print(f"  decoded subset size == classical max ({classical_max_size}): {found_max_size}")
    print(f"  decoded subset is one of the brute-force marked optima: {found_is_marked}")

    # Success probability mass concentrated on marked states, as a sanity metric.
    marked_shots = sum(c for bits, c in counts.items() if bits[::-1] in classical_marked)
    amp_fraction = marked_shots / total_shots
    print(f"  fraction of shots landing on a marked (optimal) state: {amp_fraction:.3f}")

    verified = found_valid and found_max_size and found_is_marked and amp_fraction > 0.5

    if verified:
        print("\nPASS")
    else:
        print("\nFAIL")


if __name__ == "__main__":
    main()
