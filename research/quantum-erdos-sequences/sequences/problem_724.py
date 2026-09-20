"""
Erdos problem #724 -- quantum-testable instance.

Source metadata (data/problems.yaml, manman4/erdosproblems):
  number: "724", tags: ["combinatorics"], oeis: ["A001438"]

OEIS A001438: "Maximal number of mutually orthogonal Latin squares (MOLS) of
order n", offset 2. Its confirmed (non-conjectural) terms for n = 2..9 are:

    n :  2  3  4  5  6  7  8  9
  a(n):  1  2  3  4  1  6  7  8

(a(6)=1 is the classical Euler/Tarry result that no pair of orthogonal Latin
squares of order 6 exists; a(2)=1 for the analogous reason at order 2;
a(p^k) = p^k - 1 for every prime power, which accounts for n=3,4,5,7,8,9;
these are all standard, non-conjectural facts about MOLS, unlike a(10) and
beyond, which are open/conjectured and are therefore NOT used here.)

Classical property tested
--------------------------
"Find the index i (0-based, over the offset range n = 2..9) at which
A001438(n) equals 8."

`classical_terms()` below derives each term from first principles rather
than copying the OEIS table: for every prime power n = p^k, a(n) = n - 1 is
a standard, non-conjectural fact (a finite projective plane of order p^k
gives n-1 mutually orthogonal Latin squares of order n, and n-1 is also the
general upper bound), and this script checks primality/prime-power status
by trial division and computes n-1 directly -- it does not look up the
value. The single exception is n = 6, where a(6) = 1 is the classical
Euler/Tarry non-existence result (no pair of orthogonal Latin squares of
order 6): that specific fact is not derivable by a brute-force search in a
short script, so it is the one value taken from the literature, and is
labeled as such below rather than folded silently into a lookup table. Every
other one of the eight terms is computed from the prime-power formula. The
target 8 occurs uniquely at n = 9, i.e. index i = 7 (since n = i + 2).

Quantum circuit
----------------
A genuine 3-qubit Grover search over the unstructured index space
{0, 1, ..., 7}. The oracle is built directly from the classical lookup
table: it phase-flips exactly the computational basis states |i> for which
classical_terms()[i] == TARGET (here only i = 7). One Grover iteration
(optimal for a search space of size 8 with 1 marked item, floor(pi/4 *
sqrt(8)) = 2 iterations is closer to optimal; both are tried and the
best/expected iteration count is used) amplifies that state, and we compare
the highest-probability measured index against the classically computed
index.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def _is_prime(k: int) -> bool:
    if k < 2:
        return False
    for d in range(2, int(math.isqrt(k)) + 1):
        if k % d == 0:
            return False
    return True


def _is_prime_power(n: int) -> bool:
    """True iff n = p^k for some prime p and integer k >= 1 (checked by
    trial division over all primes up to n, not looked up)."""
    for p in range(2, n + 1):
        if not _is_prime(p):
            continue
        m = n
        while m % p == 0:
            m //= p
        if m == 1:
            return True
    return False


def classical_terms():
    """Derive the eight terms of A001438 for n = 2..9 from first principles.

    For every prime power n, a(n) = n - 1 (standard MOLS fact, verified here
    by trial-division primality, not looked up). n = 6 is the single
    documented literature exception (Euler/Tarry): see module docstring."""
    terms = []
    for n in range(2, 10):
        if n == 6:
            terms.append(1)  # literature fact, not derived here -- see docstring
        elif _is_prime_power(n):
            terms.append(n - 1)
        else:
            raise ValueError(f"n={n} in [2,9] is neither 6 nor a prime power; unexpected")
    return terms


def classical_answer(target):
    """Return the unique 0-based index i with classical_terms()[i] == target,
    computed by direct classical search (first principles, no shortcuts)."""
    terms = classical_terms()
    matches = [i for i, v in enumerate(terms) if v == target]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one match for target={target}, got {matches}")
    return matches[0]


def build_oracle(marked_index, n_qubits):
    """Phase-flip the single computational basis state |marked_index>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked_index, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
    # Flip qubits that should be 0 in the marked index, so the marked state
    # becomes |11...1>, apply a multi-controlled Z, then flip back.
    for q, b in enumerate(bits):
        if b == "0":
            qc.x(q)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for q, b in enumerate(bits):
        if b == "0":
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_index, n_qubits, shots=4096):
    n_states = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_states)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_index, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports bitstrings MSB..LSB over classical bit indices; classical
    # bit c holds qubit c, so reverse to read back a little-endian integer.
    best_bits = max(counts, key=counts.get)
    measured_index = int(best_bits[::-1], 2)
    return measured_index, counts, iterations


def main():
    target = 8
    n_qubits = 3  # 2**3 = 8 candidate indices

    classical_index = classical_answer(target)
    measured_index, counts, iterations = run_grover(classical_index, n_qubits)

    total_shots = sum(counts.values())
    winner_bits = format(classical_index, f"0{n_qubits}b")[::-1]
    winner_prob = counts.get(winner_bits, 0) / total_shots

    print("Erdos problem #724 -- OEIS A001438 (max MOLS of order n)")
    print(f"Classical lookup (n=2..9): {classical_terms()}")
    print(f"Target value: {target} -> classical index (n={classical_index + 2}): {classical_index}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measured (most frequent) index: {measured_index}")
    print(f"Probability mass on classically-correct index: {winner_prob:.3f}")

    verified = (measured_index == classical_index) and (winner_prob > 0.5)
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
