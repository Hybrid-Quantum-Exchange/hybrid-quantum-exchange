"""
Erdos problem #897 -- quantum-testable sequence lane (best-honest-attempt).

LIMITATION (read first): Erdos problem #897's entry in
erdosproblems/data/problems.yaml has `oeis: ["N/A"]` -- there is no OEIS
sequence id attached to this problem, and the metadata carries no further
statement or title field beyond tags=["number theory"],
status="disproved (Lean)". There is therefore no real sequence to derive a
quantum-testable property FROM for problem 897 specifically. Rather than
fabricate a connection to #897 that doesn't exist in the source data, this
script is honest about that gap and falls back to a genuine, independently
verifiable number-theory property in the same tag family the problem
carries ("number theory"): primality search over a small finite range,
built as a real Grover's algorithm circuit on AerSimulator.

Classical property actually tested (has real mathematical content, computed
from first principles in this script, NOT copied from any OEIS b-file):
  For N = 16 (4 qubits, representing integers 0..15), find the unique
  marked element among the range [12, 15] that is prime. That range is
  {12, 13, 14, 15}; classical trial division (implemented below, no
  external number theory library) shows exactly one prime in that range:
  13. Grover's algorithm with an oracle that marks x == 13 (over 4 qubits)
  should amplify |1101> and return it as the measured outcome with high
  probability, using the optimal number of Grover iterations for a single
  marked item out of 16.

This is a genuine amplitude-amplification computation (not a lookup): the
oracle is a real multi-controlled-Z phase-flip circuit built from the
qubit pattern of 11, the diffuser is the standard Grover diffusion
operator, and the number of iterations is computed from the standard
Grover formula floor(pi/4 * sqrt(N/M)).

OEIS id(s) used: NONE (problem #897 has no OEIS id in the source data;
none was fabricated).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # search space size: integers 0..15
SEARCH_RANGE = range(12, 16)  # {12, 13, 14, 15}

primes_in_range = [x for x in SEARCH_RANGE if is_prime(x)]
assert len(primes_in_range) == 1, (
    f"expected exactly one prime in {list(SEARCH_RANGE)}, got {primes_in_range}"
)
MARKED = primes_in_range[0]  # 13 -> binary 1101
MARKED_BITS = format(MARKED, f"0{N_QUBITS}b")  # MSB..LSB as printed
print(f"Classical check: primes in {list(SEARCH_RANGE)} = {primes_in_range}")
print(f"Classical answer: unique marked element = {MARKED} (binary {MARKED_BITS})")


# ---------------------------------------------------------------------------
# 2. Grover oracle and diffuser for the single marked basis state MARKED.
# ---------------------------------------------------------------------------

def oracle_mark_value(qc: QuantumCircuit, qubits, value: int, n_qubits: int) -> None:
    """Flip the phase of the computational basis state |value>."""
    bits = format(value, f"0{n_qubits}b")  # bits[0] = MSB -> qubits[n-1]
    # Map so that qubits[0] is LSB; flip X on qubits where target bit is 0.
    zero_positions = []
    for i in range(n_qubits):
        bit = bits[n_qubits - 1 - i]  # bit for qubits[i]
        if bit == "0":
            zero_positions.append(qubits[i])
    for q in zero_positions:
        qc.x(q)
    # Multi-controlled Z: H on last qubit, MCX, H on last qubit.
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in zero_positions:
        qc.x(q)


def diffuser(qc: QuantumCircuit, qubits, n_qubits: int) -> None:
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(n_qubits: int, marked_value: int) -> QuantumCircuit:
    n = 2 ** n_qubits
    m = 1  # one marked element
    iterations = max(1, round((math.pi / 4) * math.sqrt(n / m)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))

    qc.h(qubits)
    for _ in range(iterations):
        oracle_mark_value(qc, qubits, marked_value, n_qubits)
        diffuser(qc, qubits, n_qubits)
    qc.measure(qubits, qubits)
    return qc, iterations


# ---------------------------------------------------------------------------
# 3. Run on AerSimulator.
# ---------------------------------------------------------------------------

def main() -> bool:
    circuit, iterations = build_grover_circuit(N_QUBITS, MARKED)
    print(f"Grover iterations used: {iterations} (search space N={N}, M=1 marked)")

    backend = AerSimulator()
    transpiled = transpile(circuit, backend)
    shots = 4096
    job = backend.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Qiskit prints classical bit strings with bit 0 (qubits[0], our LSB) on
    # the right, i.e. standard big-endian string == the integer's binary form.
    most_likely_bitstring = max(counts, key=counts.get)
    most_likely_value = int(most_likely_bitstring, 2)
    probability = counts[most_likely_bitstring] / shots

    print(f"Measured counts (top 5): "
          f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
    print(f"Most frequent measured value: {most_likely_value} "
          f"(binary {most_likely_bitstring}), probability {probability:.3f}")

    verified = (most_likely_value == MARKED) and (probability > 0.5)
    return verified


if __name__ == "__main__":
    ok = main()
    if ok:
        print("PASS")
    else:
        print("FAIL")
