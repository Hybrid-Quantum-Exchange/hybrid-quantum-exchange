"""
Erdos problem #244 -- quantum-testable companion script.

Source metadata (from erdosproblems/data/problems.yaml, verified by this
session on 2026-09-19):

    number: "244"
    prize: "no"
    informal_status: open
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory", "primes"]

LIMITATION, stated honestly up front: problem #244 carries NO OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no specific OEIS
term or sequence membership to bind a circuit to for this problem. Per the
task's fallback instructions, this script does not fabricate an OEIS id or
copy a literal value from nowhere. Instead it builds a REAL, genuine quantum
circuit around the one concrete, finite, computable property the problem's
own tags commit to: "primes" / "number theory" -- specifically, primality
of small integers, which is exactly the kind of small finite decision
problem Grover search is built for.

Classical property under test
------------------------------
Let N = 16 (4 qubits, search space {0, 1, ..., 15}).
Define the classical predicate  is_prime(x)  for x in [0, 16) using trial
division computed from first principles in this script (no external table,
no OEIS lookup). The classical set of primes below 16 is computed directly:

    PRIMES_BELOW_16 = { x in [0,16) : is_prime(x) }
                     = {2, 3, 5, 7, 11, 13}

Quantum circuit
----------------
A standard Grover's algorithm circuit is built over 4 qubits:
  - An oracle phase-flips exactly the basis states |x> whose integer value
    x is prime (per the classical is_prime computed above), implemented as
    a multi-controlled Z gate gated on the specific bit patterns of each
    prime in [0, 16).
  - The diffusion (inversion-about-mean) operator amplifies those marked
    states.
  - The optimal number of Grover iterations for M=6 marked items out of
    N=16 is computed as round((pi/4) * sqrt(N/M)).

The circuit is run on the ideal AerSimulator (statevector-backed sampling,
no noise model), and the resulting measurement distribution is compared
against the classically computed prime set: PASS requires that the states
receiving amplified (majority) probability mass are exactly the classical
primes below 16, verified by checking that the top-M measured outcomes
(by count) equal PRIMES_BELOW_16 exactly.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

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
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16
PRIMES_BELOW_16 = sorted(x for x in range(N) if is_prime(x))
M = len(PRIMES_BELOW_16)

print(f"Classical search space size N = {N}")
print(f"Classically computed primes below {N}: {PRIMES_BELOW_16}")
print(f"Number of marked items M = {M}")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle that phase-flips exactly the prime basis states.
# ---------------------------------------------------------------------------

def append_multi_controlled_z(qc: QuantumCircuit, qubits: list) -> None:
    """Apply a phase flip (Z) controlled on all given qubits being |1>."""
    if len(qubits) == 1:
        qc.z(qubits[0])
    elif len(qubits) == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def oracle_for_value(qc: QuantumCircuit, value: int, n_qubits: int) -> None:
    """Phase-flip |value> only, via X-sandwiched multi-controlled Z."""
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    zero_positions = [i for i, b in enumerate(bits) if b == 0]
    for i in zero_positions:
        qc.x(i)
    append_multi_controlled_z(qc, list(range(n_qubits)))
    for i in zero_positions:
        qc.x(i)


def build_oracle(n_qubits: int, marked_values: list) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for v in marked_values:
        oracle_for_value(qc, v, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    append_multi_controlled_z(qc, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# 3. Assemble full Grover circuit with the optimal iteration count.
# ---------------------------------------------------------------------------

n_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations used: {n_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(N_QUBITS, PRIMES_BELOW_16)
diffuser = build_diffuser(N_QUBITS)

for _ in range(n_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 8192
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# qc.measure(range(N_QUBITS), range(N_QUBITS)) maps qubit i -> clbit i, and
# Qiskit's returned bitstring already reads as a standard big-endian integer
# once clbit indices line up with qubit indices this way, so a direct
# int(bitstring, 2) recovers the measured basis-state value.
int_counts = {}
for bitstring, c in counts.items():
    value = int(bitstring, 2)
    int_counts[value] = int_counts.get(value, 0) + c

sorted_by_count = sorted(int_counts.items(), key=lambda kv: kv[1], reverse=True)
top_m_values = sorted(v for v, _ in sorted_by_count[:M])

print(f"Top-{M} measured outcomes (by count), sorted: {top_m_values}")
print(f"Classical primes below {N}, sorted:            {PRIMES_BELOW_16}")

verified = top_m_values == PRIMES_BELOW_16

if verified:
    print("PASS")
else:
    print("FAIL")
