"""
Erdos problem #322 (https://www.erdosproblems.com/322)
OEIS sequences used: A025456 "Number of partitions of n into 3 positive
cubes" and A025418 "Least sum of 3 positive cubes in exactly n ways"
(tags: number theory, powers).

Classical property tested (derived and checked from first principles in
this script, not copied from OEIS):

    A025456(29) == 1

i.e. 29 has EXACTLY ONE way to be written as an unordered sum of three
positive cubes a^3 + b^3 + c^3 with 1 <= a <= b <= c. A brute-force
classical search over the only cubes that can possibly contribute
(1^3=1, 2^3=8, 3^3=27, since 4^3=64 > 29) finds that the unique
partition is (a, b, c) = (1, 1, 3): 1 + 1 + 27 = 29.

Quantum approach: Grover's search.

We encode ordered triples (a, b, c) with a, b, c in {1, 2, 3, 4} using
2 qubits per variable (6 qubits total, register value v in 0..3 maps to
the real value v + 1). This is a small enough search space (64 basis
states) that the "is a^3+b^3+c^3 == 29" oracle can be built exactly, by
first finding classically (in Python, not by magic) every basis state
index whose triple sums to 29, and then compiling a phase-flip oracle
that marks exactly those computational-basis states with multi-controlled
Z gates. Grover's diffusion operator is then applied the standard optimal
number of times, and the resulting statevector should concentrate onto
the marked basis state(s).

Because A025456(29) == 1, there is exactly one ordered representative in
the encoded search space for the unordered solution (1,1,3), but since
the sum is invariant under permuting equal/unequal entries, several
*ordered* triples can realize the same unordered partition (any
permutation of (1,1,3) that stays within our 4-valued register). This
script enumerates the ORDERED marked states classically first (ground
truth), builds the Grover oracle from that ground truth, runs it on the
ideal AerSimulator, and checks that measurement concentrates on exactly
the classically-marked ordered states -- which is exactly the quantum
circuit "verifying" the OEIS-derived classical fact A025456(29) == 1
(one unordered partition, realized by a known, classically-enumerated
set of ordered triples).

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

TARGET = 29
REG_BITS = 2          # bits per variable -> values 0..3 -> real value v+1
NUM_VARS = 3
NUM_QUBITS = REG_BITS * NUM_VARS   # 6 qubits, 64 basis states


def real_value(v: int) -> int:
    return v + 1


def classical_partition_count(n: int) -> int:
    """A025456(n): number of partitions of n into 3 positive cubes
    (unordered, a <= b <= c), found by brute force over the only cubes
    that can possibly contribute."""
    max_root = int(round(n ** (1 / 3))) + 2
    count = 0
    for a in range(1, max_root + 1):
        if a ** 3 > n:
            break
        for b in range(a, max_root + 1):
            if a ** 3 + b ** 3 > n:
                break
            for c in range(b, max_root + 1):
                s = a ** 3 + b ** 3 + c ** 3
                if s == n:
                    count += 1
                elif s > n:
                    break
    return count


def find_marked_indices(n: int, reg_bits: int) -> list:
    """Classically enumerate every ordered triple (a,b,c), each register
    value in 0..2**reg_bits-1 (mapped to real value+1), whose cubes sum
    to n. Returns the list of basis-state indices (0..2**(3*reg_bits)-1),
    using index = a_bits*4^2 + b_bits*4^1 + c_bits*4^0 with a_bits etc.
    being the raw register value (0-based)."""
    dim = 2 ** reg_bits
    marked = []
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                ra, rb, rc = real_value(a), real_value(b), real_value(c)
                if ra ** 3 + rb ** 3 + rc ** 3 == n:
                    idx = a * dim * dim + b * dim + c
                    marked.append(idx)
    return marked


def build_oracle(num_qubits: int, marked_indices: list) -> QuantumCircuit:
    """Phase-flip oracle: for each marked computational basis index,
    apply X on the 0-bits, a multi-controlled Z on all qubits, then
    undo the X gates. Qubit 0 is the least-significant bit of the index."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for idx in marked_indices:
        bits = [(idx >> i) & 1 for i in range(num_qubits)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all qubits (phase flip of |11...1>)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
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
    # --- Step 1: classical ground truth ---
    a25456_29 = classical_partition_count(TARGET)
    print(f"Classical A025456({TARGET}) [# unordered partitions into 3 "
          f"positive cubes] = {a25456_29}")
    assert a25456_29 == 1, "expected exactly one unordered partition of 29"

    marked = find_marked_indices(TARGET, REG_BITS)
    marked = sorted(set(marked))
    print(f"Classically marked ordered basis states (search space size "
          f"{2 ** NUM_QUBITS}): {marked}")
    for idx in marked:
        dim = 2 ** REG_BITS
        a = idx // (dim * dim)
        b = (idx // dim) % dim
        c = idx % dim
        ra, rb, rc = real_value(a), real_value(b), real_value(c)
        print(f"  index {idx}: (a,b,c)=({ra},{rb},{rc}), "
              f"{ra}^3+{rb}^3+{rc}^3={ra**3+rb**3+rc**3}")
    assert len(marked) > 0, "no ordered triples found -- classical bug"

    # --- Step 2: build Grover circuit ---
    N = 2 ** NUM_QUBITS
    M = len(marked)
    iterations = max(1, round(math.pi / 4 * math.sqrt(N / M)))
    print(f"N={N} basis states, M={M} marked, Grover iterations={iterations}")

    oracle = build_oracle(NUM_QUBITS, marked)
    diffuser = build_diffuser(NUM_QUBITS)

    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    qc.h(range(NUM_QUBITS))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

    # --- Step 3: run on ideal AerSimulator ---
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit order: rightmost char of the bitstring is qubit 0, and
    # int(bs, 2) already treats it as the least-significant bit, so no
    # reversal is needed -- the bitstring is already qubit0-as-LSB.
    def bitstring_to_index(bs: str) -> int:
        return int(bs, 2)

    counts_by_index = {}
    for bs, c in counts.items():
        counts_by_index[bitstring_to_index(bs)] = c

    sorted_counts = sorted(counts_by_index.items(), key=lambda kv: -kv[1])
    print("Top measured basis states (index: count):")
    for idx, c in sorted_counts[:5]:
        print(f"  {idx}: {c}")

    marked_set = set(marked)
    marked_prob = sum(c for idx, c in counts_by_index.items()
                       if idx in marked_set) / shots
    top_index, top_count = sorted_counts[0]
    print(f"Total probability mass on classically-marked states: "
          f"{marked_prob:.4f}")

    # Success condition: the most-sampled outcome is one of the
    # classically-marked (verified) solutions, and the marked states
    # collectively dominate the distribution -- i.e. Grover search found
    # and amplified exactly the classical answer to
    # "A025456(29) == 1 partition, realized by triples marked above".
    quantum_result_matches_classical = (
        top_index in marked_set and marked_prob > 0.5
    )

    print()
    if quantum_result_matches_classical:
        print("PASS: Grover search on the ideal AerSimulator concentrated "
              "on the classically-verified solution(s) of "
              f"a^3+b^3+c^3={TARGET}, matching A025456({TARGET})=1.")
    else:
        print("FAIL: quantum result did not match the classical answer.")

    return quantum_result_matches_classical


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
