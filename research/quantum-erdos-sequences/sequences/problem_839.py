"""
Erdos problem #839 -- quantum-testable companion script.

Source record (data/problems.yaml, erdosproblems repo, entry "number: 839"):
    prize: no
    status: open (as of 2025-08-31)
    tags: ["number theory"]
    oeis: ["N/A"]

LIMITATION (report honestly, do not fake a match to problem 839 itself):
Problem 839 has no OEIS sequence attached in the source data (oeis: ["N/A"]),
and the repository entry carries no numeric statement beyond the tag "number
theory" -- there is no small, finite, computable property of "the sequence
for problem 839" to target, because no sequence is on record. Fabricating an
OEIS id or a property attributed to problem 839 would misrepresent the
source, so this script does not do that.

Instead, in the spirit of "identify a small, finite, computable property that
a quantum circuit can genuinely compute", this script builds a real Grover
search circuit for the classical number-theoretic property that problem 839's
own tag names -- primality -- over a small finite domain. This is offered as
an honest, clearly-labeled substitute demonstration, not as a claim about
problem 839's (nonexistent) OEIS sequence.

Property tested:
    Search space: integers n in [0, 15] (4 qubits, basis state |n>).
    Property P(n): n is prime (classically: 2, 3, 5, 7, 11, 13 -- computed
    below in Python via trial division, not hard-coded from memory).

Circuit:
    A standard Grover search (oracle + diffuser, iterated the
    theoretically-optimal number of times for this marked-fraction) is run on
    the ideal AerSimulator. The oracle phase-flips exactly the basis states
    n for which P(n) is True, using X gates to remap "prime bit patterns" onto
    an all-ones pattern that a multi-controlled Z can flip.

Verification:
    After circuit execution, the set of n with the top measured
    probabilities (as many as there are classically prime values) is compared
    against the classically computed prime set. PASS is printed iff they
    match exactly.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed here, not copied from memory/OEIS).
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


N_QUBITS = 4
DOMAIN = list(range(2 ** N_QUBITS))  # 0..15
CLASSICAL_PRIMES = sorted(n for n in DOMAIN if is_prime(n))
print(f"Classical primes in [0, {2 ** N_QUBITS - 1}]: {CLASSICAL_PRIMES}")


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffuser for the marked set CLASSICAL_PRIMES.
# ---------------------------------------------------------------------------

def build_oracle(marked_values: list[int], n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
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


import math

N = 2 ** N_QUBITS
M = len(CLASSICAL_PRIMES)
# Optimal number of Grover iterations for M marked items out of N.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"N={N}, M={M} marked, Grover iterations={iterations}")

oracle = build_oracle(CLASSICAL_PRIMES, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

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
result = backend.run(compiled, shots=4096).result()
counts = result.get_counts()

# Convert bitstrings (Qiskit: qubit N_QUBITS-1 ... qubit 0, big-endian in the
# printed key) to integers.
value_counts: dict[int, int] = {}
for bitstring, c in counts.items():
    n = int(bitstring, 2)
    value_counts[n] = value_counts.get(n, 0) + c

ranked = sorted(value_counts.items(), key=lambda kv: -kv[1])
top_m_values = sorted(n for n, _ in ranked[:M])

print(f"Top-{M} most frequent measured values: {top_m_values}")
print(f"Measured distribution (value: count): {dict(sorted(value_counts.items()))}")

verified = top_m_values == CLASSICAL_PRIMES

print("PASS" if verified else "FAIL")
