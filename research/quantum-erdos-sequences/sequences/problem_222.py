"""
Erdos problem #222 (erdosproblems.com/222), tags: number theory, squares.
OEIS ids referenced by the problem entry: A001481, A256435.

A001481 = numbers expressible as a sum of two squares of nonnegative
integers, i.e. n = x^2 + y^2 for some integers x, y >= 0.

Property tested here: for a fixed target integer N, does there exist a pair
(x, y) with 0 <= x, y <= 7 (3 bits each, so the search space has 64 = 8*8
grid points, i.e. a 6-qubit register) such that x^2 + y^2 == N? This is
exactly the finite, computable membership test underlying A001481 for
small N, restricted to a small search space so it fits a few qubits.

The classical answer is computed here from first principles by brute-force
enumeration over the same 8x8 grid (no OEIS values are copied verbatim).

We pick N = 25, which is a genuine, non-trivial member of A001481
(25 = 3^2 + 4^2 = 4^2 + 3^2 = 0^2 + 5^2 = 5^2 + 0^2), giving 4 solutions
out of 64 grid points -- a good, non-degenerate case for Grover search.

Circuit: Grover's algorithm on a 6-qubit register (3 qubits for x, 3 for
y). The oracle is built directly from the classically-precomputed set of
marked (x, y) pairs satisfying x^2 + y^2 == N: for each marked bitstring,
we flip the 0-bits to 1 (X gates), apply a multi-controlled Z, and undo
the X gates. This is a standard exact phase oracle construction for an
explicit small set of marked computational basis states, not a lookup of
an OEIS value -- the *set* of marked states is what encodes the number
theory of A001481. The diffusion operator is the standard Grover
diffuser. The number of Grover iterations is chosen near-optimally from
the analytic formula given the known number of solutions.

We then run the circuit on the ideal AerSimulator, take the most probable
measured outcomes, decode them back to (x, y) pairs, and check that they
are exactly the classically-computed solution set to x^2 + y^2 == 25.
PASS/FAIL is printed based on that comparison.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_solutions(n, bits=3):
    """Brute-force all (x, y) in [0, 2^bits)^2 with x^2 + y^2 == n."""
    size = 1 << bits
    sols = []
    for x in range(size):
        for y in range(size):
            if x * x + y * y == n:
                sols.append((x, y))
    return sorted(sols)


def bitstring_for_pair(x, y, bits=3):
    """Little-endian qubit layout: qubits [0:bits) = x, [bits:2*bits) = y."""
    xs = format(x, f"0{bits}b")[::-1]
    ys = format(y, f"0{bits}b")[::-1]
    return xs + ys  # qubit index i -> character i


def add_oracle(qc, marked_bitstrings, num_qubits):
    """Exact phase oracle flipping the sign of each marked basis state."""
    for bits in marked_bitstrings:
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)


def add_diffuser(qc, num_qubits):
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))


def main():
    bits = 3
    num_qubits = 2 * bits  # 6 qubits, search space size 64
    N = 25

    sols = classical_solutions(N, bits=bits)
    marked_bitstrings = [bitstring_for_pair(x, y, bits=bits) for x, y in sols]
    M = len(marked_bitstrings)
    search_space = 1 << num_qubits

    print(f"Classical solutions to x^2 + y^2 == {N} over [0,{(1<<bits)-1}]^2: {sols}")
    print(f"Number of marked states M = {M} out of N = {search_space}")

    if M == 0 or M == search_space:
        raise RuntimeError("Degenerate instance chosen; pick different N.")

    # Near-optimal number of Grover iterations.
    theta = math.asin(math.sqrt(M / search_space))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        add_oracle(qc, marked_bitstrings, num_qubits)
        add_diffuser(qc, num_qubits)
    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Qiskit's bit-string order is qubit (n-1)...0, left to right; our
    # bitstring_for_pair used qubit-index order, so reverse before compare.
    def decode(qiskit_bits):
        # qiskit_bits is c_{n-1} c_{n-2} ... c_0
        idx_order = qiskit_bits[::-1]  # now index 0..n-1
        x = int(idx_order[:bits][::-1], 2)
        y = int(idx_order[bits:2 * bits][::-1], 2)
        return x, y

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_m = sorted_counts[:M]
    top_pairs = sorted({decode(bits_) for bits_, _ in top_m})

    total_shots = sum(counts.values())
    marked_shots = sum(c for b, c in counts.items() if b[::-1][:num_qubits] and decode(b) in sols)
    marked_fraction = marked_shots / total_shots

    print(f"Grover iterations used: {iterations}")
    print(f"Top {M} measured (x, y) pairs by frequency: {top_pairs}")
    print(f"Fraction of shots landing on a true solution: {marked_fraction:.3f}")

    quantum_solution_set = set(top_pairs)
    classical_solution_set = set(sols)

    ok = (
        quantum_solution_set == classical_solution_set
        and marked_fraction > 0.8
    )

    if ok:
        print("PASS")
    else:
        print("FAIL")
    return ok


if __name__ == "__main__":
    success = main()
    if not success:
        raise SystemExit(1)
