"""
Erdos problem #393 -- quantum-testable instance.

OEIS sequence used: A388302.
  a(n) = the smallest m >= 1 such that n! = b_1 * b_2 * ... * b_t, with
  b_1 < b_2 < ... < b_t (t >= 2 distinct increasing integer factors), and
  m = b_t - b_1 (the spread between the largest and smallest factor).
  (Confirmed from the OEIS entry: "a(n) = is the smallest m >= 1 such that
  n! = b_1*...*b_t with b_1 < ... < b_t and m = b_t - b_1.")

Property tested here (finite, small, computable):
  Restricted to the two-factor case (t = 2), for n = 3 we have 3! = 6.
  The only way to write 6 = d * e with 1 <= d < e is d = 2, e = 3 (since
  d = 1, e = 6 gives spread 5, which is worse), so the minimum two-factor
  spread is min(e - d) = 1, achieved uniquely at d = 2.
  This classical minimum (1, achieved at d = 2) is exactly a(3) = 1 as
  listed in OEIS A388302 (first term of the sequence, n=3 -> 1), so the
  two-factor case is not merely a toy: it reproduces the true sequence
  value for n = 3.

  The classical answer is (re)computed from first principles in
  `classical_solve()` below, by brute-force trial division over all
  d = 1..6, with no OEIS values hard-coded.

Quantum approach:
  A genuine Grover search (amplitude amplification) over a 3-qubit
  register representing candidate divisors d in {0, ..., 7} (6 fits in
  3 bits; states 6, 7 are simply never marked). The oracle -- built
  directly from the classically precomputed valid-divisor set, not
  smuggling in the raw OEIS term -- marks exactly the states d for which
  d divides 6, d < 6/d, and (6/d - d) equals the true minimum spread
  found by classical_solve(). For n = 3 there is a unique marked state,
  d = 2, so Grover's algorithm with the standard optimal number of
  iterations should recover it with high probability. We run the ideal
  AerSimulator, take the most frequent measured bitstring, decode it,
  and compare it against the independently computed classical answer.

Limitation: this instance only searches the restricted two-factor (t=2)
sub-problem of A388302's definition (not the full multi-factor search
over all factorizations of n!), because implementing an in-circuit
arbitrary-factorization oracle for larger n is not a small circuit.
For n = 3, the two-factor restriction happens to coincide with the true
optimum, so the classical answer used for comparison is the real a(3).
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_solve(n: int):
    """Brute-force, first-principles computation of the two-factor
    minimum spread of n!: min over 1 <= d < e, d*e == n! of (e - d).

    Returns (min_spread, best_d, best_e).
    """
    target = math.factorial(n)
    best = None
    best_d = None
    best_e = None
    for d in range(1, target + 1):
        if d * d > target:
            break
        if target % d == 0:
            e = target // d
            if d < e:
                spread = e - d
                if best is None or spread < best:
                    best = spread
                    best_d = d
                    best_e = e
    return best, best_d, best_e


def build_marked_set(n: int, num_qubits: int):
    """Return the set of integers d (0..2**num_qubits - 1) that should be
    marked by the Grover oracle: valid two-factor divisors of n! whose
    spread equals the classical minimum spread."""
    target = math.factorial(n)
    min_spread, _, _ = classical_solve(n)
    marked = set()
    limit = 2 ** num_qubits
    for d in range(1, min(target, limit)):
        if target % d == 0:
            e = target // d
            if d < e and (e - d) == min_spread:
                marked.add(d)
    return marked


def grover_oracle(num_qubits: int, marked_states: set) -> QuantumCircuit:
    """Phase-flip oracle marking each state in marked_states."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(n: int, num_qubits: int = 3, shots: int = 2048):
    marked_states = build_marked_set(n, num_qubits)
    if not marked_states:
        raise RuntimeError(f"No marked states found for n={n}; instance is empty.")

    N = 2 ** num_qubits
    M = len(marked_states)
    # Optimal number of Grover iterations for M marked items out of N.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = grover_oracle(num_qubits, marked_states)
    diff = diffuser(num_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diff, inplace=True)

    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Most frequent measured bitstring -> decode to integer (little-endian
    # qubit ordering as used above, so reverse before int()).
    best_bitstring = max(counts, key=counts.get)
    measured_value = int(best_bitstring[::-1], 2)

    return measured_value, counts, marked_states, iterations


def main():
    n = 3
    num_qubits = 3  # covers d in 0..7, enough since target divisors <= 6

    min_spread, best_d, best_e = classical_solve(n)
    print(f"Erdos problem #393 / OEIS A388302, n = {n}, n! = {math.factorial(n)}")
    print(
        f"Classical (first-principles) two-factor minimum spread: "
        f"{min_spread} achieved at d = {best_d}, e = {best_e}"
    )

    measured_value, counts, marked_states, iterations = run_grover(n, num_qubits)
    print(f"Grover iterations used: {iterations}")
    print(f"Oracle-marked states (candidate divisors): {sorted(marked_states)}")
    print(f"Measurement counts: {counts}")
    print(f"Most frequent measured value (decoded): {measured_value}")

    classical_answer = best_d
    quantum_answer = measured_value

    passed = quantum_answer == classical_answer
    print(f"Classical answer (d achieving min spread): {classical_answer}")
    print(f"Quantum (Grover) answer: {quantum_answer}")
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
