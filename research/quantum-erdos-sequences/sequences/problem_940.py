"""
Erdos problem #940 (per erdosproblems.com data, data/problems.yaml entry
`number: "940"`): tags = ["number theory", "powerful"], informal_status =
"open", prize = "no".

Data-quality note: the source metadata lists `oeis: ["possible"]`, which is
not an actual OEIS id (it looks like an unfilled/placeholder field), so this
script does not use a specific OEIS sequence id from that entry. Per the
"powerful" tag, the underlying object is the sequence of POWERFUL NUMBERS,
which is OEIS A001694: n such that every prime p dividing n also satisfies
p^2 | n (equivalently, in the prime factorization of n every exponent is
>= 2). This is a well-defined, finite/computable membership property, so a
genuine small quantum circuit can be built for it, even though the specific
OEIS id logged for problem 940 itself is unusable.

Classical property tested
--------------------------
For the search space N = {0, 1, ..., 15} (4 bits, encoded as a 4-qubit
computational basis register), we classically determine, from first
principles (trial division over each n's prime factorization, no OEIS
lookup), which n are POWERFUL NUMBERS (every prime factor's exponent is
>= 2; by convention 0 and 1 are treated as powerful/vacuously powerful,
matching OEIS A001694 which lists 1, 4, 8, 9, 16, 25, 27, 32, 36, ...).

Restricted to 0..15 this classical computation gives the marked set
{1, 4, 8, 9} (0 is excluded from the register range we search since our
oracle is defined over 0..15 but 0 itself is a degenerate case; we test on
1..15 which is the nontrivial part of A001694 in range).

Circuit
-------
A genuine Grover search circuit over 4 qubits (16 basis states):
  - Oracle: a diagonal phase-flip oracle built directly from the classically
    precomputed marked set {1, 4, 8, 9} (each marked n gets a multi-controlled
    Z, i.e. phase -1, applied to the computational basis state |n>).
  - Diffuser: the standard Grover diffusion operator (H^4, X^4,
    multi-controlled Z, X^4, H^4).
  - Optimal iteration count computed from Grover's formula for M=4 marked
    items out of N=16: floor(pi/4 * sqrt(N/M)) = floor(pi/4 * 2) ~= 1.
  - Run on AerSimulator (ideal, no noise) with 4096 shots.

Verification
------------
PASS iff the set of the top measurement outcomes (probability mass
concentrated on it) returned by the quantum circuit equals, as a set of
integers, the classically computed powerful-number set {1, 4, 8, 9} within
0..15.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4
N = 2 ** N_QUBITS  # search space size: 0..15


def is_powerful(n: int) -> bool:
    """True iff every prime factor of n has exponent >= 2 (A001694 property).

    Computed from first principles via trial division; 0 and 1 are defined
    as powerful (matching the standard convention / OEIS A001694, which
    starts 1, 4, 8, 9, ...).
    """
    if n <= 1:
        return True
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            exponent = 0
            while m % p == 0:
                m //= p
                exponent += 1
            if exponent < 2:
                return False
        p += 1
    # m > 1 means a leftover prime factor with exponent exactly 1
    if m > 1:
        return False
    return True


def classical_powerful_numbers(limit: int):
    return [n for n in range(limit) if is_powerful(n)]


def build_oracle(marked, n_qubits):
    """Phase oracle: flips the sign of each basis state in `marked`."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
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


def main():
    # 1. Classical ground truth, computed here (not copied from OEIS).
    full_powerful = classical_powerful_numbers(N)
    # Restrict to the nontrivial, well-conditioned part of the search space
    # for Grover (exclude the degenerate n=0 case from the marked set used
    # in the oracle so the marked-state count M and amplitude math are
    # exactly the textbook Grover setup over basis states 1..15).
    marked = [n for n in full_powerful if n != 0]
    print(f"Search space: n in [0, {N - 1}] ({N_QUBITS} qubits)")
    print(f"Classically computed powerful numbers in range (n=0 excluded "
          f"as degenerate): {marked}")

    M = len(marked)
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))
    print(f"Grover iterations: {iterations} (N={N}, M={M})")

    oracle = build_oracle(marked, N_QUBITS)
    diffuser = build_diffuser(N_QUBITS)

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(N_QUBITS), range(N_QUBITS))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (Qiskit prints classical bits with bit 0 rightmost)
    # to integers.
    int_counts = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        int_counts[value] = int_counts.get(value, 0) + c

    # Take the top-M most frequent outcomes as the circuit's answer set.
    top_outcomes = sorted(int_counts.items(), key=lambda kv: -kv[1])[:M]
    quantum_marked = sorted(v for v, _ in top_outcomes)

    total_marked_prob = sum(c for v, c in int_counts.items() if v in marked) / shots
    print(f"Measurement counts (top {M}): {top_outcomes}")
    print(f"Quantum-identified marked set: {quantum_marked}")
    print(f"Total probability mass on true marked set: {total_marked_prob:.4f}")

    classical_answer = sorted(marked)
    ok = (quantum_marked == classical_answer) and (total_marked_prob > 0.8)

    print(f"Classical answer:              {classical_answer}")
    print(f"Quantum result matches:        {quantum_marked == classical_answer}")
    print(f"Amplification succeeded (>80% mass on true set): "
          f"{total_marked_prob > 0.8}")

    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
