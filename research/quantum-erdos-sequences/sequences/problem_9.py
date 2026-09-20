"""
Erdos problem #9 -- quantum-testable instance.

Source: /home/user/manman4/erdosproblems/data/problems.yaml, entry "number: '9'"
  oeis: ["A006286"]
  tags: ["number theory", "additive basis", "primes"]

Erdos problem #9 concerns de Polignac's conjecture: is every sufficiently
large odd number expressible as the sum of a prime and a power of two
(n = p + 2^k)?  OEIS A006286 lists the odd numbers that are NOT of this
form -- i.e. counterexamples to de Polignac's conjecture (Erdos/Crocker
studied constructions producing infinitely many such n; whether they are
the *only* ones is open, matching the "open" informal_status).

Classical property tested here (computed from first principles below, not
copied from OEIS):
    For n = 125 (odd, chosen small), does there exist k in {0,...,7} with
    2^k < n such that n - 2^k is prime?
    If such k exists, n is a sum of a prime and a power of two, i.e. n is
    NOT a member of A006286 (this is the "well-behaved" case); the search
    for a witnessing k is exactly the kind of small finite search a Grover
    circuit can perform.

This script:
  1. Classically enumerates k = 0..7, computes n - 2^k, and tests
     primality by trial division (first principles, no external libs).
     This yields the classical answer: the set of "good" k values.
  2. Builds a 3-qubit Grover search circuit whose oracle marks exactly the
     classically-determined good k values (a real amplitude-amplification
     oracle + diffuser, iterated the Grover-optimal number of times for
     the size of the marked set), runs it on AerSimulator, and takes the
     most-frequently-measured outcome as the quantum answer.
  3. Compares the quantum-found k against the classical set of good k and
     prints PASS/FAIL.

Instance size: N = 8 (3 qubits) -- well within "N <= ~64, few qubits".
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import math


def is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    i = 3
    while i * i <= v:
        if v % i == 0:
            return False
        i += 2
    return True


def classical_good_k(n: int, num_k: int = 8):
    """Return the set of k in [0, num_k) with n - 2**k prime (and > 0)."""
    good = set()
    for k in range(num_k):
        v = n - 2 ** k
        if v > 0 and is_prime(v):
            good.add(k)
    return good


def build_oracle(good_k, n_qubits=3):
    """Phase-flip oracle marking each basis state in `good_k` (little-endian
    integer encoding of the 3 qubits)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for k in good_k:
        bits = format(k, f"0{n_qubits}b")[::-1]  # little-endian per-qubit
        # flip qubits that are 0 in this k so the target pattern becomes |111>
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits=3):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_search(good_k, n_qubits=3, shots=2000):
    N = 2 ** n_qubits
    M = len(good_k)
    if M == 0:
        raise ValueError("no marked states -- Grover search is not meaningful")

    # Grover-optimal iteration count for M marked out of N states.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(good_k, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    n = 125
    num_k = 8  # 3 qubits, 2^0 .. 2^7 (2^7=128 > n so it is naturally excluded)

    good_k = classical_good_k(n, num_k)
    print(f"n = {n}")
    print(f"Classical search over k in [0,{num_k}): n - 2^k prime for k in {sorted(good_k)}")
    for k in sorted(good_k):
        print(f"  witness: {n} - 2^{k} = {n - 2**k} (prime)")

    counts = grover_search(good_k, n_qubits=3, shots=2000)
    print(f"Grover measurement counts: {counts}")

    # Most frequent measured bitstring -> integer k (little-endian qubit order,
    # Qiskit returns bitstrings MSB-first over qubit index n-1..0).
    best_bitstring = max(counts, key=counts.get)
    # Qiskit prints bitstrings as q_{n-1}...q_0; qubit i holds bit i of k,
    # so reading the string as ordinary binary gives k directly.
    measured_k = int(best_bitstring, 2)

    print(f"Quantum-found k (most frequent outcome): {measured_k}")

    # Verify: the quantum answer must be one of the classically-determined
    # good k values, and the marked set must be non-trivially amplified
    # (marked states should dominate the measurement counts).
    marked_shots = sum(v for bstr, v in counts.items() if int(bstr, 2) in good_k)
    total_shots = sum(counts.values())
    amplification_ok = marked_shots / total_shots > 0.5  # amplified well above 2/8=0.25 baseline

    verified = (measured_k in good_k) and amplification_ok

    print(f"Marked-state fraction of shots: {marked_shots}/{total_shots} = {marked_shots/total_shots:.3f}")
    print(f"Classical good k set: {sorted(good_k)}")
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
