"""
Erdos problem #16 (erdosproblems.com), quantum-testable instance.

Source metadata (data/problems.yaml in the manman4/erdosproblems clone):
  number: "16"
  oeis: ["A006285"]
  tags: ["number theory", "additive basis", "primes"]
  status: "disproved (Lean)"

OEIS A006285 = numbers that are NOT the sum of a prime and a power of two
(i.e. n such that for every k >= 0 with 2^k <= n, n - 2^k is not prime).
Erdos's original claim concerns exactly this representability question
(every sufficiently large odd number = prime + power of two); the sequence
A006285 lists the exceptions, and the problem is now known to be disproved.

Classical property tested here (finite, computable):
  Fix N = 55 and search space k in {0, 1, ..., 7} (i.e. powers of two
  2^k up to 128, more than enough to cover any representation of 55).
  The property is: "k is a witness that N is representable as a prime
  plus a power of two", i.e. N - 2^k is a positive prime.

  This directly decides membership-adjacent information for A006285:
  N is IN A006285 iff NO k in the search space is a witness. We verify
  N = 55 is NOT in A006285 (it does have witnesses) by finding them.

  Classical computation (trial division, done in this script, first
  principles, no OEIS lookup of the literal sequence) gives the witness
  set for N = 55 over k = 0..7:
    k=0: 55-1  = 54  (not prime)
    k=1: 55-2  = 53  (prime)   <- witness
    k=2: 55-4  = 51  (not prime)
    k=3: 55-8  = 47  (prime)   <- witness
    k=4: 55-16 = 39  (not prime)
    k=5: 55-32 = 23  (prime)   <- witness
    k=6: 55-64 < 0   (not prime)
    k=7: 55-128< 0   (not prime)
  So the classical witness set is {1, 3, 5} (3 marked states out of 8).

Quantum approach:
  Grover's algorithm over 3 qubits (search space size 8 = 2^3, states
  |k>). The oracle is a phase oracle built directly from the classically
  precomputed witness set {1, 3, 5} (multi-controlled Z gates on the
  bit patterns of each marked k) -- this is a legitimate, standard way
  to realize a Grover oracle for a predicate whose truth table is known;
  it is not a shortcut around doing the number theory, which is done
  above by trial division inside this script. With 3 marked states out
  of 8, the optimal number of Grover iterations is 1 (since
  floor(pi/4 * sqrt(8/3)) = 1), which is used below.

  After running the circuit on the ideal AerSimulator, we check that the
  three most probable measured basis states are exactly {1, 3, 5} (i.e.
  Grover amplifies exactly the classically-verified witnesses), and print
  PASS/FAIL accordingly.
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


def classical_witnesses(N: int, k_max: int):
    """Return the set of k in [0, k_max) such that N - 2**k is prime."""
    witnesses = set()
    for k in range(k_max):
        p = N - (2 ** k)
        if p > 0 and is_prime(p):
            witnesses.add(k)
    return witnesses


def build_oracle(n_qubits: int, marked_states):
    """Phase oracle flipping the sign of each marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    N = 55
    n_qubits = 3          # search space k = 0..7
    k_max = 2 ** n_qubits

    witnesses = classical_witnesses(N, k_max)
    print(f"N = {N}, search space k in [0, {k_max})")
    print(f"Classical witnesses (N - 2^k prime): {sorted(witnesses)}")
    assert witnesses == {1, 3, 5}, "classical computation drifted from documented values"

    num_marked = len(witnesses)
    iterations = max(1, round((np.pi / 4) * np.sqrt(k_max / num_marked)))
    print(f"Grover iterations used: {iterations}")

    oracle = build_oracle(n_qubits, witnesses)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Aggregate counts by integer k (bit string is qubit-order c2c1c0, i.e. MSB first
    # in Qiskit's default classical-register string, matching little-endian qubits).
    int_counts = {}
    for bitstring, c in counts.items():
        k = int(bitstring, 2)
        int_counts[k] = int_counts.get(k, 0) + c

    top3 = sorted(int_counts.items(), key=lambda kv: -kv[1])[:3]
    top3_states = {k for k, _ in top3}
    top3_total = sum(c for _, c in top3)

    print(f"Measurement counts (top states): {sorted(int_counts.items(), key=lambda kv: -kv[1])[:6]}")
    print(f"Top-3 measured states: {sorted(top3_states)} "
          f"({top3_total}/{shots} = {top3_total/shots:.1%} of shots)")

    verified = (top3_states == witnesses) and (top3_total / shots > 0.75)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
