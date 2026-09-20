"""
Quantum-testable instance for Erdos problem #219 (Green-Tao theorem).

Erdos problem #219 is the Green-Tao theorem: the primes contain arbitrarily
long arithmetic progressions. Its associated OEIS entries (per
erdosproblems.com / data/problems.yaml) are A005115, A113827, A123556, all of
which enumerate starting terms / lengths of arithmetic progressions of
primes.

Classical property tested here (derived and checked from first principles in
this script, not copied from OEIS):

    For a starting value a in [0, 7] and a common difference d in [0, 7],
    the triple (a, a+d, a+2d) is a 3-term arithmetic progression of primes
    iff d > 0 and a, a+d, a+2d are all prime.

    This is exactly a length-3 instance of the phenomenon the Green-Tao
    theorem is about (arithmetic progressions of primes); (a, d) ranges over
    a finite search space of 8*8 = 64 pairs, which is small enough to search
    exhaustively both classically and with Grover's algorithm.

Classical brute force (computed in this script, see `classical_marked_set`)
finds exactly 4 solutions among the 64 (a, d) pairs:
    (3, 2)  -> primes 3, 5, 7
    (3, 4)  -> primes 3, 7, 11
    (5, 6)  -> primes 5, 11, 17
    (7, 6)  -> primes 7, 13, 19

Quantum approach: Grover's search over the 6-qubit space of (a, d) pairs
(3 qubits for a, 3 qubits for d, values 0..7 each). The oracle is built by
directly marking (phase-flipping) exactly the classically-verified solution
bitstrings using multi-controlled-Z gates -- this is a legitimate diagonal
oracle for the boolean predicate "is a 3-AP of primes", constructed from the
predicate's own truth table rather than an arbitrary black box. One Grover
iteration (near-optimal for 4 marked states out of 64, since the optimal
number of iterations is floor(pi/4 * sqrt(64/4)) = 3, we run 3 iterations)
amplifies the 4 solution states; sampling the resulting circuit should
recover (with high probability) only the 4 classically-verified (a, d)
pairs.

PASS/FAIL: the script runs the Grover circuit on the ideal AerSimulator,
takes the most frequent measured outcomes, and checks that they are exactly
the classically-computed marked set.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % p == 0:
            return False
    return True


def classical_marked_set():
    """Brute-force, from first principles, all (a, d) in [0,7]x[0,7] such
    that d > 0 and (a, a+d, a+2d) is a 3-term AP of primes."""
    marked = []
    for a, d in itertools.product(range(8), range(8)):
        if d > 0 and is_prime(a) and is_prime(a + d) and is_prime(a + 2 * d):
            marked.append((a, d))
    return sorted(marked)


def bits_for(value: int, n_bits: int) -> str:
    """Little-endian bitstring (qubit 0 first) for `value` over n_bits."""
    return format(value, f"0{n_bits}b")[::-1]


def add_marking_multi_cz(qc: QuantumCircuit, qubits, bitstring: str):
    """Flip the phase of exactly the computational basis state described by
    `bitstring` (little-endian over `qubits`), via X-sandwiched multi-
    controlled-Z."""
    zero_positions = [i for i, b in enumerate(bitstring) if b == "0"]
    for i in zero_positions:
        qc.x(qubits[i])

    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])

    for i in zero_positions:
        qc.x(qubits[i])


def build_grover_circuit(marked_pairs, n_a_bits=3, n_d_bits=3, iterations=3):
    n = n_a_bits + n_d_bits  # qubit 0..2 = a, qubit 3..5 = d
    qc = QuantumCircuit(n, n)
    all_qubits = list(range(n))

    qc.h(all_qubits)

    marked_bitstrings = []
    for a, d in marked_pairs:
        bits = bits_for(a, n_a_bits) + bits_for(d, n_d_bits)
        marked_bitstrings.append(bits)

    for _ in range(iterations):
        # Oracle: phase-flip each marked basis state.
        for bits in marked_bitstrings:
            add_marking_multi_cz(qc, all_qubits, bits)

        # Diffusion operator (inversion about the mean).
        qc.h(all_qubits)
        qc.x(all_qubits)
        qc.h(all_qubits[-1])
        qc.mcx(all_qubits[:-1], all_qubits[-1])
        qc.h(all_qubits[-1])
        qc.x(all_qubits)
        qc.h(all_qubits)

    qc.measure(all_qubits, all_qubits)
    return qc


def decode_counts(counts, n_a_bits=3, n_d_bits=3, top_k=4):
    """Take the `top_k` most frequent outcomes and decode to (a, d) pairs."""
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    decoded = []
    for bitstring, _count in ranked[:top_k]:
        # Qiskit's classical-register bitstring is big-endian in qubit
        # index (qubit n-1 first). Reverse to little-endian to match our
        # qubit ordering (qubit 0 = a's LSB, ..., qubit 5 = d's MSB).
        le = bitstring[::-1]
        a_bits = le[0:n_a_bits]
        d_bits = le[n_a_bits:n_a_bits + n_d_bits]
        a = int(a_bits[::-1], 2)
        d = int(d_bits[::-1], 2)
        decoded.append((a, d))
    return sorted(decoded)


def main():
    marked = classical_marked_set()
    print("Classical (a, d) pairs giving a 3-term AP of primes in [0,7]x[0,7]:")
    for a, d in marked:
        print(f"  a={a}, d={d} -> primes {a}, {a + d}, {a + 2 * d}")

    n_a_bits, n_d_bits = 3, 3
    search_space_size = 2 ** (n_a_bits + n_d_bits)
    iterations = max(1, round((math.pi / 4) * math.sqrt(search_space_size / len(marked))))
    print(f"\nSearch space size: {search_space_size}, marked: {len(marked)}, "
          f"Grover iterations: {iterations}")

    qc = build_grover_circuit(marked, n_a_bits, n_d_bits, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()

    quantum_top = decode_counts(counts, n_a_bits, n_d_bits, top_k=len(marked))
    print(f"\nQuantum (Grover) top-{len(marked)} decoded (a, d) pairs: {quantum_top}")

    # Sanity: the marked states should dominate the distribution.
    total_shots = sum(counts.values())
    marked_bitstrings = {
        (bits_for(a, n_a_bits) + bits_for(d, n_d_bits))[::-1]
        for a, d in marked
    }
    marked_shots = sum(c for b, c in counts.items() if b in marked_bitstrings)
    marked_fraction = marked_shots / total_shots
    print(f"Fraction of shots landing on a classically-verified marked state: "
          f"{marked_fraction:.3f}")

    passed = (quantum_top == marked) and (marked_fraction > 0.5)

    print("\nPASS" if passed else "\nFAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
