"""
Erdos problem #850 -- the Erdos-Woods conjecture (OEIS A343101, "Erdos-Woods
numbers": those k for which there exists a length-(k+1) run of consecutive
integers not determined, up to translation, by the set of prime divisors of
each of its members). The known smallest Erdos-Woods number is k = 16.

Erdos-Woods numbers are fundamentally about consecutive integers sharing
prime-factor structure across a fixed offset. This script tests a small,
finite, honestly-computed instance of exactly that kind of structure using
the specific offset k = 16 that anchors A343101:

    Classical property under test:
        Search n in {1, ..., 63} (6 bits) for values where
            rad(n) == rad(n + 16)
        where rad(m) is the squarefree kernel (radical) of m -- the product
        of the distinct primes dividing m. Equal radicals across the +16
        offset is exactly the kind of "same prime divisors" coincidence
        that the Erdos-Woods property studies (whether an interval's
        divisor pattern can recur elsewhere).

    This is NOT a claim that these n solve the Erdos-Woods problem itself
    (that requires checking a full interval of length 17 against ALL other
    intervals, which is not a small finite computation); it is a small,
    genuinely computable arithmetic search directly in the spirit of the
    sequence's defining mechanism, with the correct answer computed here
    classically from first principles before the quantum circuit is run.

Approach: brute-force the classical set S = {n in [1,63] : rad(n) = rad(n+16)}.
Then run Grover's algorithm on 6 qubits with an oracle built directly from S
(a multi-controlled-Z marking exactly the n in S, no shortcuts), using the
textbook-optimal number of Grover iterations, on the ideal AerSimulator.
The circuit is judged PASS if the most frequently measured 6-bit string,
interpreted as an integer, is a member of S.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


N_QUBITS = 6
N = 1 << N_QUBITS  # 64
OFFSET = 16


def radical(m: int) -> int:
    """Product of the distinct prime factors of m (m >= 1)."""
    if m <= 1:
        return 1
    r = 1
    x = m
    p = 2
    while p * p <= x:
        if x % p == 0:
            r *= p
            while x % p == 0:
                x //= p
        p += 1 if p == 2 else 2
    if x > 1:
        r *= x
    return r


def classical_solution_set():
    """n in [1, 63] with rad(n) == rad(n + 16), computed from first principles."""
    sols = []
    for n in range(1, N):
        if radical(n) == radical(n + OFFSET):
            sols.append(n)
    return sols


def build_oracle(marked, n_qubits):
    """Phase oracle flipping the sign of exactly the |n> states in `marked`."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # Multi-controlled Z on all n_qubits marking |111...1> (post X-flips
        # this is exactly the target basis state m).
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    marked = classical_solution_set()
    print(f"Classical search space: n in [1, {N - 1}], offset k = {OFFSET}")
    print(f"Classical solution set S = {{n : rad(n) = rad(n+{OFFSET})}}: {marked}")
    assert len(marked) > 0, "classical search found no solutions; cannot build oracle"

    M = len(marked)
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"|S| = {M} out of N = {N}; using {iterations} Grover iteration(s)")

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))

    oracle = build_oracle(marked, N_QUBITS)
    diffuser = build_diffuser(N_QUBITS)
    for _ in range(iterations):
        qc.compose(oracle, range(N_QUBITS), inplace=True)
        qc.compose(diffuser, range(N_QUBITS), inplace=True)

    qc.measure(range(N_QUBITS), range(N_QUBITS))

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit prints classical bits as c[n-1]...c[0] (qubit 0 is the
    # rightmost/least-significant character), which is exactly standard
    # binary place-value order for the integer n -- no reversal needed.
    def bits_to_int(bitstring):
        return int(bitstring, 2)

    top_bits, top_count = max(counts.items(), key=lambda kv: kv[1])
    top_n = bits_to_int(top_bits)
    hit_prob = sum(c for b, c in counts.items() if bits_to_int(b) in marked) / shots

    print(f"Most frequent measured n = {top_n} (count {top_count}/{shots})")
    print(f"Probability mass on classical solution set S: {hit_prob:.3f}")

    passed = (top_n in marked) and (hit_prob > 0.5)
    if passed:
        print("PASS")
    else:
        print("FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
