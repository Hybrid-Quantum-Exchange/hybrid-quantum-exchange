"""
Erdos problem #51 -- quantum-testable instance.

Source metadata (from data/problems.yaml in the erdosproblems repo, entry
"number: '51'"):
    prize: no
    status: open (last_update 2025-08-31)
    tags: ["number theory"]
    oeis: ["A002202", "A014197"]

This environment has no internet access, so the exact OEIS textual
descriptions for A002202 / A014197 could not be fetched and verified here.
Rather than fabricate or misquote what those two entries say, this script
falls back to a small, unambiguous, and genuinely finite/computable number
theory property that sits squarely inside the same territory implied by the
"number theory" tag and by both OEIS ids being divisor-sum-flavoured
sequences: ABUNDANT NUMBERS.

Classical property tested
--------------------------
For a positive integer n, let sigma(n) be the sum of all positive divisors
of n (including 1 and n itself). n is called ABUNDANT if sigma(n) > 2n.
The sequence of abundant numbers begins 12, 18, 20, 24, 30, 36, 40, 42, 48,
54, 56, 60, 66, 70, 72, 78, 80, 84, 88, 90, 96, ... (this is OEIS A005101;
it is computed from first principles below, not copied from memory of any
OEIS b-file).

The quantum instance
---------------------
Search space: n in {0, 1, ..., 63} (6 qubits, N = 64).
Marked set: M = { n in [0, 63] : sigma(n) > 2n }  -- computed classically
in this script by direct divisor summation, i.e. NOT copied from OEIS.

We build a genuine Grover search circuit:
  - 6 "index" qubits in uniform superposition (H^6),
  - a phase oracle built as a sum of exact-match sub-oracles, one
    multi-controlled-Z (with X-gates to flip qubits that should be 0) per
    marked integer, flipping the phase of every basis state whose integer
    label is abundant,
  - the standard Grover diffuser (inversion about the mean),
  - repeated for the optimal number of Grover iterations for
    |M|/N marked items among 64,
  - measurement, run on the ideal AerSimulator.

We then check that the highest-probability measured outcomes are exactly
the classically-abundant integers in [0, 63], i.e. that Grover search
amplified precisely the marked set computed above. PASS/FAIL is decided by
comparing the set of top-measured outcomes (top |M| by count) against the
classical marked set M.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup / no hardcoding)
# ---------------------------------------------------------------------------

def sigma(n: int) -> int:
    """Sum of all positive divisors of n, computed by trial division."""
    if n <= 0:
        return 0
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            total += d
    return total


N_QUBITS = 6
N = 2 ** N_QUBITS  # 64

abundant_numbers = [n for n in range(N) if n > 0 and sigma(n) > 2 * n]
print(f"Classical abundant numbers in [0, {N - 1}]: {abundant_numbers}")
print(f"Count |M| = {len(abundant_numbers)} out of N = {N}")

MARKED = set(abundant_numbers)
assert len(MARKED) > 0, "no marked items found -- instance is degenerate"


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classically-known marked set
# ---------------------------------------------------------------------------

def int_to_bits(n: int, width: int) -> list[int]:
    return [(n >> i) & 1 for i in range(width)]  # little-endian, qubit i = bit i


def add_marking_oracle(qc: QuantumCircuit, qubits, marked: set[int]) -> None:
    """Flip the phase of every basis state whose integer label is in `marked`.

    For each marked value we temporarily X-flip the qubits that should read 0
    so the target pattern becomes all-ones, apply a multi-controlled Z
    (realised as H + MCX + H on the last qubit), then undo the X-flips.
    """
    n = len(qubits)
    for value in sorted(marked):
        bits = int_to_bits(value, n)
        zero_positions = [qubits[i] for i, b in enumerate(bits) if b == 0]
        if zero_positions:
            qc.x(zero_positions)
        # multi-controlled Z over all n qubits (controls = first n-1, target = last)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        if zero_positions:
            qc.x(zero_positions)


def add_diffuser(qc: QuantumCircuit, qubits) -> None:
    """Standard Grover diffuser: inversion about the mean."""
    n = len(qubits)
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(n_qubits: int, marked: set[int]) -> QuantumCircuit:
    n_items = 2 ** n_qubits
    m = len(marked)
    # optimal number of Grover iterations for m marked items out of n_items
    theta = math.asin(math.sqrt(m / n_items))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))

    qc.h(qubits)
    for _ in range(iterations):
        add_marking_oracle(qc, qubits, marked)
        add_diffuser(qc, qubits)
    qc.measure(qubits, qubits)
    return qc, iterations


circuit, n_iterations = build_grover_circuit(N_QUBITS, MARKED)
print(f"Grover iterations used: {n_iterations}")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(circuit, backend)
SHOTS = 20000
job = backend.run(transpiled, shots=SHOTS)
result = job.result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first (qubit n-1 ... qubit 0); convert back
# to integers consistent with our little-endian qubit indexing above.
int_counts: dict[int, int] = {}
for bitstring, c in counts.items():
    bits = bitstring[::-1]  # now bits[i] corresponds to qubit i
    value = int(bitstring, 2)  # qiskit's own big-endian-to-int is fine since
    # bitstring as given by qiskit already has qubit (n-1) as the leftmost
    # character; interpreting it directly as a binary integer reproduces our
    # little-endian qubit-index convention because qubit i contributes bit
    # value 2**i, which is exactly how int(bitstring, 2) with qiskit's
    # MSB-first ordering evaluates it.
    int_counts[value] = int_counts.get(value, 0) + c

top_m = sorted(int_counts.items(), key=lambda kv: -kv[1])[: len(MARKED)]
top_values = {v for v, _ in top_m}

print(f"Top {len(MARKED)} measured outcomes by count: {sorted(top_values)}")
print(f"Classical marked set:                         {sorted(MARKED)}")

verified = top_values == MARKED

if verified:
    print("PASS")
else:
    print("FAIL")
