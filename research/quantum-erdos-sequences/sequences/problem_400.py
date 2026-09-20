"""
Erdos problem #400 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems clone):
  number: "400"
  tags: ["number theory", "factorials"]
  oeis: ["possible"]

Limitation, stated honestly: problem #400's yaml entry does not carry a real
OEIS sequence id -- the field is the literal placeholder string "possible",
not an id of the form A0xxxxx. There is therefore no citable OEIS sequence to
build a "membership in the sequence" oracle from for this problem. This
script does not fabricate an OEIS id. Instead, honoring the entry's actual
tags ("number theory", "factorials"), it builds a genuine, self-contained
finite/computable factorial-related number-theory property -- the classic
Brocard-style question of which small n make n! + 1 prime -- and runs a real
Grover search over that property. The classical answer is derived from
scratch in this script (trial-division primality test), not copied from any
table, and the quantum circuit is graded against that from-scratch answer.

Property tested:
  For n in {1, ..., 8} (indices 0..7, 3 qubits), is n! + 1 prime?
  This is a small, finite, exactly computable predicate (factorial +
  primality test), matching the "factorials" / "number theory" tags.

Classical answer (computed below, from first principles):
  n=1: 1!+1=2   -> prime  (marked)
  n=2: 2!+1=3   -> prime  (marked)
  n=3: 3!+1=7   -> prime  (marked)
  n=4: 4!+1=25  -> not prime (5*5)
  n=5: 5!+1=121 -> not prime (11*11)
  n=6: 6!+1=721 -> not prime (7*103)
  n=7: 7!+1=5041-> not prime (71*71)
  n=8: 8!+1=40321 -> not prime (23*1753)
  Marked set (0-indexed n-1): {0, 1, 2}

Quantum approach:
  A 3-qubit Grover search over the 8 basis states |n-1>, with a phase oracle
  that flips the sign of exactly the marked basis states determined above,
  followed by the standard diffuser, run for the optimal number of Grover
  iterations. The ideal AerSimulator statevector/measurement result should
  concentrate amplitude (and hence measurement counts) on the marked set
  {0,1,2} (i.e. n in {1,2,3}).

Pass criterion:
  The three most-measured basis states (by count, out of 4096 shots) on the
  ideal simulator must equal exactly the classically-derived marked set.
"""

import math
import itertools

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


def is_prime(k: int) -> bool:
    """From-scratch trial division primality test."""
    if k < 2:
        return False
    if k in (2, 3):
        return True
    if k % 2 == 0:
        return False
    i = 3
    while i * i <= k:
        if k % i == 0:
            return False
        i += 2
    return True


def classical_marked_set(n_values):
    """Compute, from first principles, which n give n! + 1 prime."""
    marked = []
    for idx, n in enumerate(n_values):
        val = math.factorial(n) + 1
        if is_prime(val):
            marked.append(idx)
    return marked


def build_oracle(num_qubits: int, marked_indices):
    """Phase oracle flipping the sign of each marked computational basis state."""
    qc = QuantumCircuit(num_qubits, name="Oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{num_qubits}b")
        # Flip qubits that are 0 in this index so the target pattern becomes
        # all-ones, apply a multi-controlled Z, then flip back.
        zero_positions = [i for i, b in enumerate(reversed(bits)) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        elif num_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser(num_qubits: int):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    elif num_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def main():
    n_values = list(range(1, 9))  # n = 1..8 -> indices 0..7, 3 qubits
    num_qubits = 3
    assert 2 ** num_qubits == len(n_values)

    marked = classical_marked_set(n_values)
    print("n values tested:", n_values)
    print("classical marked indices (n-1) where n!+1 is prime:", marked)
    print("i.e. n in", [n_values[i] for i in marked])

    N = 2 ** num_qubits
    M = len(marked)
    # Optimal number of Grover iterations for N items, M marked.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(num_qubits, marked)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # counts keys are bitstrings 'q2q1q0' (Qiskit little-endian); convert to int.
    int_counts = {}
    for bitstring, c in counts.items():
        idx = int(bitstring, 2)
        int_counts[idx] = int_counts.get(idx, 0) + c

    print("measurement counts (index -> count):", int_counts)

    top_indices = sorted(int_counts, key=lambda k: int_counts[k], reverse=True)[:M]
    top_indices_sorted = sorted(top_indices)
    marked_sorted = sorted(marked)

    print("top", M, "measured indices:", top_indices_sorted)
    print("classical marked indices:      ", marked_sorted)

    verified = top_indices_sorted == marked_sorted
    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
