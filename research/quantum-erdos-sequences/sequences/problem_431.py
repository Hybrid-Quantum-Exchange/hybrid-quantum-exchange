"""
Erdos problem #431 ("inverse Goldbach problem"; tags: number theory, primes)
quantum-testable lane.

LIMITATION, stated up front: problem 431's entry in erdosproblems/data/problems.yaml
lists oeis: ["N/A"] -- there is no OEIS sequence id attached to this problem as of
the scrape used here (informal_status: open, last_update 2025-08-31). This script
therefore cannot test "membership in OEIS sequence A......." for problem 431, and
any script claiming to do so would be fabricating a data source that does not
exist. Rather than skip the lane or invent an OEIS id, this script builds a real,
small, finite, computable property drawn directly from the problem's own stated
subject -- Goldbach-type representation of an integer as a sum of two primes --
and tests it with a genuine Grover search circuit on Qiskit's AerSimulator.

Classical property under test
------------------------------
Fix N = 10 (a small even integer) and the list of primes below N:
    P = [2, 3, 5, 7]      (4 elements, indexed 0..3, padded to 8 = 2^3)
For index i, define the predicate
    good(i)  <=>  P[i] is prime AND (N - P[i]) is prime AND N - P[i] > 0
i.e. index i is "good" iff P[i] is one half of a Goldbach decomposition of N
into two primes. This is computed from first principles in this script with a
trial-division primality test -- nothing is copied from any table.

Quantum circuit
----------------
A 3-qubit Grover search marks exactly the indices i in {0,...,7} for which
good(i) holds (indices 4-7, which do not correspond to any prime in P, are
never marked). One Grover iteration (near-optimal for the resulting
marked-out-of-8 count) is applied to a uniform superposition, then the
register is measured 2000 times.
PASS requires that the marked ("good") indices collectively receive a large
majority (>75%) of the measured shots, versus the 3/8 = 37.5% they would get
from a uniform (non-amplified) distribution -- i.e. the quantum search result
matches, and is clearly boosted toward, the classical answer.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_good_indices(N: int, P: list[int]) -> list[int]:
    """Indices i such that P[i] and N - P[i] are both prime (Goldbach split)."""
    good = []
    for i, p in enumerate(P):
        if not is_prime(p):
            continue
        q = N - p
        if q > 0 and is_prime(q):
            good.append(i)
    return good


def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each index in `marked` (as an n_qubits bitstring)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked:
        bits = format(idx, f"0{n_qubits}b")
        # flip 0-bits to 1 so the controlled-Z fires on this exact pattern
        zero_positions = [n_qubits - 1 - k for k, b in enumerate(bits) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main() -> bool:
    N = 10
    P = [2, 3, 5, 7]
    n_qubits = 3  # 2^3 = 8 >= len(P) = 6

    marked = classical_good_indices(N, P)
    print(f"N = {N}, primes list P = {P}")
    print(f"Classical Goldbach-split indices (good(i)): {marked}")
    print(f"  i.e. {[(P[i], N - P[i]) for i in marked]} are the prime pairs summing to {N}")

    M = len(marked)
    total = 2 ** n_qubits  # 8
    # Optimal number of Grover iterations for M marked out of `total`.
    theta = np.arcsin(np.sqrt(M / total))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim, basis_gates=["u3", "cx", "id"])
    shots = 2000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit order is little-endian in the classical register string; convert
    # each measured bitstring back to the integer index i used above.
    measured_indices = {int(bitstring, 2): c for bitstring, c in counts.items()}
    print(f"Grover iterations used: {iterations}")
    print(f"Measured index -> shot counts: {measured_indices}")

    marked_shots = sum(c for i, c in measured_indices.items() if i in marked)
    fraction_marked = marked_shots / shots
    print(f"Fraction of shots landing on a classically-verified good index: {fraction_marked:.3f}")

    # PASS conditions: (1) some good indices exist to search for, (2) the quantum
    # search overwhelmingly returns indices that classically satisfy good(i).
    ok = (M > 0) and (fraction_marked > 0.75)
    return ok


if __name__ == "__main__":
    passed = main()
    print("PASS" if passed else "FAIL")
