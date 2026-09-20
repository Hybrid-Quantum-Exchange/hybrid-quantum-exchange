"""
Erdos problem #979  (https://www.erdosproblems.com/979)
OEIS: A385316 - "Smallest number that is the sum of 3 cubes of primes in
exactly n different ways."  a(1) = 24 = 2^3 + 2^3 + 2^3.

Classical property tested
--------------------------
Restrict to the four smallest primes P = [2, 3, 5, 7].  Consider every
ordered triple (i, j, k) in {0,1,2,3}^3 (encoded as a 6-bit string, two
bits per index) and the condition

    P[i]^3 + P[j]^3 + P[k]^3 == 24

The script first checks this classically, by brute force over all 4^3 = 64
triples, and finds that there is EXACTLY ONE ordered triple satisfying it:
(i, j, k) = (0, 0, 0), i.e. 2^3 + 2^3 + 2^3 = 24.  This reproduces, on a
small finite instance, the defining fact behind a(1) = 24 in A385316: 24 is
expressible as a sum of three prime cubes in exactly one way.

Quantum computation
--------------------
A genuine Grover search circuit is built over the 6-qubit space of all 64
ordered triples (i, j, k).  The oracle is a diagonal phase-flip oracle that
marks exactly the basis state(s) satisfying P[i]^3+P[j]^3+P[k]^3 == 24 -
built directly from the classical brute-force search above (not hard-coded
from OEIS), i.e. the oracle marks whichever computational basis states turn
out to satisfy the arithmetic condition. Because there is exactly one
marked state out of N = 64, the optimal number of Grover iterations is
round(pi/4 * sqrt(64)) = 6, after which measurement should recover the
marked triple (0,0,0) with high probability.

The script runs this circuit on the ideal AerSimulator and checks that the
most frequently measured bitstring decodes to the same (i, j, k) = (0,0,0)
found by the classical brute-force search, and that its measured
probability is much larger than 1/64 (i.e. genuine amplitude amplification
occurred, not a uniform random guess).

No OEIS terms are copied verbatim without verification: a(1)=24 and its
representation 2^3+2^3+2^3 are both recomputed from scratch here.
"""

import math
from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


PRIMES = [2, 3, 5, 7]      # small primes used for the finite search space
TARGET = 24                # a(1) of A385316: sum of 3 prime cubes, 1 way
NUM_INDEX_QUBITS = 2        # 2 bits -> indices 0..3 into PRIMES
NUM_QUBITS = 3 * NUM_INDEX_QUBITS  # 6 qubits total


def classical_brute_force():
    """Brute-force all ordered triples (i,j,k) in {0,1,2,3}^3 and return
    the list of index-triples whose prime cubes sum to TARGET."""
    solutions = []
    for i, j, k in product(range(4), repeat=3):
        s = PRIMES[i] ** 3 + PRIMES[j] ** 3 + PRIMES[k] ** 3
        if s == TARGET:
            solutions.append((i, j, k))
    return solutions


def triple_to_bits(i, j, k):
    """Encode (i,j,k), each in 0..3, as a 6-character bitstring
    q5 q4 | q3 q2 | q1 q0  (qiskit bit order, little-endian in the string
    printed by qiskit is reversed; we build/interpret consistently below)."""
    bits = f"{i:02b}{j:02b}{k:02b}"
    return bits


def build_oracle(marked_bits_list):
    """Diagonal phase-flip oracle marking exactly the given 6-bit strings.

    For each marked bitstring, flip the 0-bits with X, apply a
    multi-controlled Z (phase flip on |11...1>), then undo the X flips.
    This never touches the classical arithmetic itself - it is a pure
    phase oracle over 6 qubits built from the marked-state list.
    """
    qc = QuantumCircuit(NUM_QUBITS, name="oracle")
    for bits in marked_bits_list:
        # bits[0] corresponds to qubit NUM_QUBITS-1 ... bits[-1] to qubit 0
        zero_positions = [q for q, b in enumerate(reversed(bits)) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        qc.h(NUM_QUBITS - 1)
        qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
        qc.h(NUM_QUBITS - 1)
        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser():
    qc = QuantumCircuit(NUM_QUBITS, name="diffuser")
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))
    return qc


def build_grover_circuit(marked_bits_list, iterations):
    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    qc.h(range(NUM_QUBITS))

    oracle = build_oracle(marked_bits_list).to_gate()
    diffuser = build_diffuser().to_gate()

    for _ in range(iterations):
        qc.append(oracle, range(NUM_QUBITS))
        qc.append(diffuser, range(NUM_QUBITS))

    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))
    return qc


def bits_to_triple(bitstring):
    """Inverse of triple_to_bits, given a qiskit measurement bitstring
    (which is printed MSB-first, matching our (i,j,k) ordering here since
    we constructed 'bits' the same way)."""
    i = int(bitstring[0:2], 2)
    j = int(bitstring[2:4], 2)
    k = int(bitstring[4:6], 2)
    return (i, j, k)


def main():
    # --- classical computation (ground truth) ---
    solutions = classical_brute_force()
    assert len(solutions) == 1, (
        f"expected exactly one representation of {TARGET} as a sum of "
        f"three cubes from {PRIMES}, found {solutions}"
    )
    classical_answer = solutions[0]
    i, j, k = classical_answer
    check = PRIMES[i] ** 3 + PRIMES[j] ** 3 + PRIMES[k] ** 3
    assert check == TARGET
    print(f"Classical brute force over {PRIMES!r}^3 triples:")
    print(f"  unique solution (i,j,k) = {classical_answer} "
          f"-> {PRIMES[i]}^3 + {PRIMES[j]}^3 + {PRIMES[k]}^3 = {TARGET}")
    print(f"  (this reproduces a(1) = 24 of OEIS A385316 for Erdos #979: "
          f"24 = 2^3+2^3+2^3, in exactly one way)")

    marked_bits = [triple_to_bits(*classical_answer)]
    N = 2 ** NUM_QUBITS
    M = len(marked_bits)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
    print(f"\nBuilding Grover circuit: N={N} basis states, M={M} marked, "
          f"iterations={iterations}")

    qc = build_grover_circuit(marked_bits, iterations)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    top_bitstring = max(counts, key=counts.get)
    top_count = counts[top_bitstring]
    top_prob = top_count / shots
    quantum_triple = bits_to_triple(top_bitstring)

    print(f"\nMost frequent measurement: {top_bitstring} "
          f"(count {top_count}/{shots}, prob {top_prob:.3f})")
    print(f"Decoded index triple: {quantum_triple}")
    print(f"Baseline uniform-random probability per state: {1 / N:.4f}")

    verified = (
        quantum_triple == classical_answer
        and top_prob > 5 * (1 / N)  # far above uniform-random baseline
    )

    print()
    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
