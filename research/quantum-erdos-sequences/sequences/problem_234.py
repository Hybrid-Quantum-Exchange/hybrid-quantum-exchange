"""
Erdos problem #234 (https://www.erdosproblems.com/234) — quantum-testable lane.

Metadata (from data/problems.yaml in the read-only erdosproblems clone,
manman4/erdosproblems, entry "number: '234'"):
    prize: no
    informal_status: open (last_update 2025-08-31)
    tags: ["number theory", "primes"]
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #234 has no associated OEIS
sequence id (oeis: ["N/A"] in the source data). There is therefore no
"sequence membership" property of *this* problem to quantum-test directly,
and the problem itself (an open number-theory/primes conjecture of Erdos) is
not a small finite decidable statement a toy circuit could evaluate. Per the
task's fallback instructions, this script instead makes its best honest
attempt at a genuine, finite, computable property drawn from the problem's
own tags ("number theory", "primes"): primality of small integers.

Chosen property: for the integers 0..15 (4 qubits), which ones are prime?
    PRIMES(0..15) = {2, 3, 5, 7, 11, 13}
This is computed classically in this script from first principles (trial
division), independently of any lookup table, and then verified with a real
Grover search circuit built on Qiskit + AerSimulator: the oracle marks
exactly the prime computational basis states among |0..15>, and Grover
amplification is run for the optimal number of iterations for a 6-out-of-16
search. The circuit's output distribution is compared against the classical
primality set: PASS requires that the highest-probability measured states
are exactly the classical prime set.

This is a real (if modest) instance of quantum search over a mathematically
meaningful predicate connected to problem #234's own tags, not a fabricated
stand-in for a specific OEIS term — because no such term exists for this
problem.
"""

import math
from collections import Counter

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
N = 2 ** N_QUBITS  # 16
CLASSICAL_PRIMES = sorted(n for n in range(N) if is_prime(n))
print(f"Classical primes in 0..{N - 1}: {CLASSICAL_PRIMES}")
assert CLASSICAL_PRIMES == [2, 3, 5, 7, 11, 13]

M = len(CLASSICAL_PRIMES)  # number of marked states = 6


# ---------------------------------------------------------------------------
# 2. Grover oracle: phase-flip exactly the marked (prime) basis states.
# ---------------------------------------------------------------------------
def build_oracle(marked_values, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
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


def grover_iterations(n_states, n_marked):
    theta = math.asin(math.sqrt(n_marked / n_states))
    return max(1, round((math.pi / (4 * theta)) - 0.5))


oracle = build_oracle(CLASSICAL_PRIMES, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)
iterations = grover_iterations(N, M)
print(f"Grover iterations for {M} marked states out of {N}: {iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------
backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 20000
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bitstrings are printed as c[n-1]...c[0] with qubit 0 as the
# rightmost character, which is exactly the standard binary reading order
# for int(bitstring, 2) given measure(range, range) maps qubit i -> bit i.
totals = Counter()
for bitstring, cnt in counts.items():
    value = int(bitstring, 2)
    totals[value] += cnt

top_states = sorted(totals, key=lambda v: -totals[v])[:M]
top_states_sorted = sorted(top_states)

print("Measurement totals per value (0..15):")
for v in range(N):
    print(f"  {v:2d}: {totals.get(v, 0):5d} shots"
          f"{'  <- classical prime' if v in CLASSICAL_PRIMES else ''}")

print(f"Top-{M} most measured states: {top_states_sorted}")
print(f"Classical prime set:          {CLASSICAL_PRIMES}")

verified = top_states_sorted == CLASSICAL_PRIMES
print("PASS" if verified else "FAIL")
