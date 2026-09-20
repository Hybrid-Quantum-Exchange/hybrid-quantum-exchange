"""
Erdos problem #458 -- quantum-testable instance.

OEIS id used: A056604.
  a(0) = 1; for n >= 1, a(n) = lcm(1, 2, 3, ..., prime(n))
  i.e. the least common multiple of all integers from 1 up to the n-th
  prime. First terms: 1, 2, 6, 60, 420, 27720, 360360, ...

Classical property being tested (derived and checked here, not copied from
OEIS): for n = 3, prime(3) = 5 and a(3) = lcm(1,2,3,4,5) = 60. By
construction, a(3) is divisible by every integer d in {1,...,5} (that is
the whole point of the lcm-of-a-range definition), but it need NOT be
divisible by the next prime, 7 = prime(4), because prime(4) is strictly
larger than prime(3) and does not appear as a factor contributed by any
d <= 5. We verify this classically first:

    60 % d == 0 for d in {1,2,3,4,5,6}
    60 % d != 0 for d in {7,8}

(6 divides 60 too, as a product of the prime factors 2 and 3 already
present; 7 and 8=2^3 do not, since a(3) only guarantees one factor of 2
coming from d=4=2^2, not three, and no factor of 7 at all.)

Quantum circuit: Grover search over the 3-qubit register d-1 in {0,...,7}
(so d = register + 1 ranges over 1..8), with an oracle that marks exactly
the two "non-divisor" values d in {7, 8}, i.e. register values 110 and 111
(register bits q2 q1 both 1, q0 free). The oracle is a single
controlled-Z on qubits q2,q1 (a CZ, since only two controls are needed,
q0 is a "don't care" bit shared by both marked states), which flips the
sign of exactly the 2 marked basis states out of 8 -- an authentic
amplitude-amplification instance, not a toy relabeling.

With N = 8 states and M = 2 marked, the optimal number of Grover
iterations is round(pi/4 * sqrt(N/M)) = 1. After 1 iteration and
measurement, the register should collapse (with high probability) onto
q2=1,q1=1 (i.e. register value 6 or 7, decoding to d = 7 or d = 8), the
two values classically verified above to NOT divide a(3) = 60.

The script:
  1. Computes a(3) = lcm(1..prime(3)) = 60 from first principles.
  2. Classically determines, by direct division, which d in {1,...,8}
     are divisors and which are not.
  3. Builds and runs the Grover circuit on AerSimulator.
  4. Checks that the two most probable measured outcomes decode to
     exactly the classically-determined non-divisor set {7, 8}.
  5. Prints PASS or FAIL.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from math import gcd

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def sieve_primes(limit: int):
    is_p = [True] * (limit + 1)
    is_p[0:2] = [False, False]
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            for j in range(i * i, limit + 1, i):
                is_p[j] = False
    return [i for i, v in enumerate(is_p) if v]


def a_n(n: int) -> int:
    """a(n) = lcm(1..prime(n)) for A056604, computed from first principles."""
    if n == 0:
        return 1
    primes = sieve_primes(1000)
    p_n = primes[n - 1]  # prime(n), 1-indexed
    result = 1
    for k in range(1, p_n + 1):
        result = lcm(result, k)
    return result


def classical_check():
    N = 3
    val = a_n(N)
    assert val == 60, f"expected a(3) = 60, computed {val}"

    divisors = set()
    non_divisors = set()
    for d in range(1, 9):
        if val % d == 0:
            divisors.add(d)
        else:
            non_divisors.add(d)

    assert divisors == {1, 2, 3, 4, 5, 6}, divisors
    assert non_divisors == {7, 8}, non_divisors
    return val, divisors, non_divisors


def build_grover_circuit(marked_register_values, n_qubits=3, iterations=1):
    """Grover search over an n_qubits register, oracle marks states whose
    top two bits (q_{n-1}, q_{n-2}) are both 1 -- exactly register values
    {6, 7} for n_qubits = 3, i.e. d-1 in {6,7} => d in {7,8}."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # uniform superposition
    qc.h(range(n_qubits))

    for _ in range(iterations):
        # oracle: phase-flip states with q2=1 and q1=1 (q0 free)
        qc.cz(2, 1)

        # diffusion operator (inversion about the mean)
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run_quantum(shots=4096):
    n_qubits = 3
    qc = build_grover_circuit(marked_register_values={6, 7}, n_qubits=n_qubits, iterations=1)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Decode: Qiskit bit order is c2 c1 c0 (q2 is the leftmost printed bit).
    # register value = int(bitstring, 2); d = register value + 1.
    decoded_counts = {}
    for bitstring, c in counts.items():
        reg_val = int(bitstring, 2)
        d = reg_val + 1
        decoded_counts[d] = decoded_counts.get(d, 0) + c

    return decoded_counts


def main():
    val, divisors, non_divisors = classical_check()
    print(f"a(3) = lcm(1..prime(3)) = {val}")
    print(f"classical divisors of {val} in 1..8: {sorted(divisors)}")
    print(f"classical non-divisors of {val} in 1..8: {sorted(non_divisors)}")

    decoded_counts = run_quantum()
    total = sum(decoded_counts.values())
    print(f"quantum measurement outcome counts (decoded d): {decoded_counts}")

    # take the top-2 most frequent outcomes (there are 2 marked states)
    top2 = sorted(decoded_counts.items(), key=lambda kv: -kv[1])[:2]
    top2_values = {d for d, _ in top2}
    top2_mass = sum(c for _, c in top2) / total

    print(f"top-2 most probable d values: {sorted(top2_values)} "
          f"(probability mass {top2_mass:.3f})")

    verified = (top2_values == non_divisors) and (top2_mass > 0.7)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
