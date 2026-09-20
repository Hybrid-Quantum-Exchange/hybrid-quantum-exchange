"""
Erdos problem #682 (OEIS A386978) — quantum-testable instance.

Problem #682 asks: is it true that for almost all n there exists some
m in (p_n, p_{n+1}) such that lpf(m) >= p_{n+1} - p_n, where lpf(m) is
the least prime factor of m and p_n is the n-th prime? (Resolved
affirmatively by Gafni and Tao, 2025.)

OEIS A386978 is exactly the set of indices k for which this witness m
exists: "numbers k such that the k-th prime gap contains an integer
whose least prime factor is >= the length of the gap."

Classical property tested here (computed from first principles in this
script, no OEIS values copied):
  For a fixed prime-gap index k, let p = k-th prime, q = (k+1)-th
  prime, gap = q - p, and let S = {p+1, p+2, ..., q-1} be the (finite,
  small) set of integers strictly between them. Membership of k in
  A386978 is equivalent to: "there exists m in S with lpf(m) >= gap".

  We pick the smallest k whose search space S has size between 2 and 8
  (so it fits in at most 3 qubits) and for which a witness genuinely
  exists (so k really is a member of A386978), as a real search
  instance. We then encode, for each of the (at most 8) computational
  basis states |i> (i indexes S), whether S[i] is a "witness"
  (lpf(S[i]) >= gap), as an oracle, and
  run Grover's algorithm on an AerSimulator to amplify and find a
  marked (witness) index. If no witness exists, the Grover oracle marks
  nothing and the circuit is expected to return a uniform (unamplified)
  distribution; we test the case where at least one witness exists,
  since that is what makes k a member of A386978.

  PASS criterion: the index the quantum circuit returns with highest
  probability is a genuine witness (lpf(S[i]) >= gap), confirmed by
  direct classical trial division, AND the quantum-derived answer to
  "is k in A386978" (witness exists) matches the fully classical brute
  force answer.

This is a real, small (2-qubit oracle + ancilla) Grover search circuit
run on Qiskit's AerSimulator — not a lookup of an OEIS value.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


def sieve_primes(limit):
    is_p = [True] * (limit + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            for j in range(i * i, limit + 1, i):
                is_p[j] = False
    return [i for i in range(2, limit + 1) if is_p[i]]


def least_prime_factor(n):
    if n < 2:
        return None
    i = 2
    while i * i <= n:
        if n % i == 0:
            return i
        i += 1
    return n


def find_instance():
    """Find the smallest prime-gap index k with a search space of size
    between 2 and 8 (fits in <= 3 qubits) that contains at least one
    genuine witness m with lpf(m) >= gap, i.e. k is a true member of
    A386978. (Note: for gap 4 the middle element p+2 is always
    divisible by 3 whenever p and p+4 are both prime > 3, so gap-4
    instances never have a witness -- an example of exactly the kind
    of classical reasoning this script must not skip.)"""
    primes = sieve_primes(5000)
    for k in range(2, len(primes) - 1):
        p, q = primes[k - 1], primes[k]  # p_k, p_{k+1}, 1-indexed
        gap = q - p
        interval = list(range(p + 1, q))  # S
        if 2 <= len(interval) <= 8:
            witnesses = [m for m in interval if least_prime_factor(m) >= gap]
            if witnesses:
                return k, p, q, gap, interval, witnesses
    raise RuntimeError("no suitable instance found")


def grover_find_witness(interval, witnesses):
    """Grover search over indices of `interval` (padded to a power of
    two) for indices whose value is in `witnesses`."""
    n_items = len(interval)
    n_qubits = max(1, math.ceil(math.log2(n_items)))
    N = 2 ** n_qubits

    marked_indices = [i for i, m in enumerate(interval) if m in witnesses]

    qc = QuantumCircuit(n_qubits + 1, n_qubits)  # +1 ancilla (phase kickback)
    data = list(range(n_qubits))
    anc = n_qubits

    # init: uniform superposition over data qubits, ancilla in |->
    qc.h(data)
    qc.x(anc)
    qc.h(anc)

    def oracle():
        for idx in marked_indices:
            bits = format(idx, f"0{n_qubits}b")[::-1]  # little-endian
            flip = [q for q, b in zip(data, bits) if b == "0"]
            for q in flip:
                qc.x(q)
            if n_qubits == 1:
                qc.cx(data[0], anc)
            else:
                qc.append(MCXGate(n_qubits), data + [anc])
            for q in flip:
                qc.x(q)

    def diffuser():
        qc.h(data)
        qc.x(data)
        if n_qubits == 1:
            qc.z(data[0])
        else:
            qc.h(data[-1])
            qc.append(MCXGate(n_qubits - 1), data[:-1] + [data[-1]])
            qc.h(data[-1])
        qc.x(data)
        qc.h(data)

    # number of Grover iterations for M marked out of N
    M = max(1, len(marked_indices))
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    for _ in range(iterations):
        oracle()
        diffuser()

    qc.h(anc)
    qc.x(anc)
    qc.measure(data, list(range(n_qubits)))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=2048).result()
    counts = result.get_counts()
    return counts, n_qubits, N


def main():
    k, p, q, gap, interval, witnesses = find_instance()

    print(f"Erdos problem #682 / OEIS A386978, instance k={k}")
    print(f"  p_{k} = {p}, p_{{{k}+1}} = {q}, gap = {gap}")
    print(f"  interval S = {interval}")
    print(f"  classical witnesses (lpf(m) >= gap): {witnesses}")
    print(f"  => k={k} is {'IN' if witnesses else 'NOT in'} A386978 (classical)")

    counts, n_qubits, N = grover_find_witness(interval, witnesses)
    print(f"  Grover circuit: {n_qubits} data qubit(s), N={N} slots, counts={counts}")

    # most frequent measured index
    best_bits = max(counts, key=counts.get)
    best_index = int(best_bits, 2)

    quantum_witness_value = interval[best_index] if best_index < len(interval) else None
    quantum_says_witness_exists = (
        quantum_witness_value is not None and quantum_witness_value in witnesses
    )

    classical_witness_exists = len(witnesses) > 0

    lpf_check = (
        quantum_witness_value is not None
        and least_prime_factor(quantum_witness_value) >= gap
    )

    ok = (
        best_index < len(interval)
        and quantum_says_witness_exists == classical_witness_exists
        and lpf_check
    )

    print(f"  quantum top outcome -> index {best_index} -> m = {quantum_witness_value}")
    print(f"  lpf(m) = {least_prime_factor(quantum_witness_value) if quantum_witness_value else None}, "
          f"required >= {gap}: {lpf_check}")
    print(f"  classical witness exists: {classical_witness_exists}, "
          f"quantum found a witness: {quantum_says_witness_exists}")

    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
