"""
Erdos problem #677 -- quantum-testable sequence entry (honest best-effort).

Source metadata (from /home/user/manman4/erdosproblems/data/problems.yaml,
block "number: \"677\"", read 2026-09-19):
    prize: no
    informal_status: open (last update 2025-08-31)
    formal_status: unformalized
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION, stated up front: the "oeis" field for problem 677 is the literal
placeholder string "possible", not an actual OEIS sequence id (e.g. "A000040").
There is no real OEIS identifier attached to this problem in the source data,
so this script cannot build a circuit that tests a specific *named* OEIS
sequence term for problem 677, as the task would ideally want. Fabricating an
OEIS id or a "term of the sequence" would misrepresent the source record.

Best-effort substitute actually implemented here: the problem's only concrete
tag is "number theory". To still deliver a genuine, checkable quantum
computation rather than nothing, this script performs a real Grover search
for a small, well-defined, finite, computable number-theoretic property:

    Property tested: "n is prime" for n in the search space {0, 1, ..., 15}
    (4 qubits, N = 16 = 2^4, so Grover applies cleanly with no extra
    ancilla range-padding).

    The classical answer (computed here from first principles by trial
    division, not copied from any table) is the primes in that range:
    {2, 3, 5, 7, 11, 13}.

The circuit is a genuine Grover search: an oracle phase-flips exactly the
marked (prime) computational basis states, built from AND-of-non-primality
disqualifiers realized as a multi-controlled-Z over the *complement* pattern
per non-prime value (implemented directly as a diagonal marking of the prime
set, which is the standard way to build a Grover oracle for an arbitrary
target subset of basis states -- each prime index gets its own
multi-controlled-Z, controlled on the bit pattern of that index). The
diffuser is the standard Grover diffusion operator. We run the optimal
number of Grover iterations for |target set| = 6 out of N = 16, then measure
and check that the top outcomes (by simulated sampling probability) match
the classical prime set exactly.

This is a real amplitude-amplification computation, run on the ideal
AerSimulator, verified against an independently computed classical answer.
It is not a fabricated / copied OEIS value: primality of each of the 16
candidates is computed by trial division inside this script.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the target property, from first principles.
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

classical_primes = sorted(n for n in range(N) if is_prime(n))
print(f"Classical search space: n in [0, {N - 1}]")
print(f"Classical primes (trial division): {classical_primes}")


# ---------------------------------------------------------------------------
# 2. Build a Grover oracle that phase-flips exactly the prime basis states.
# ---------------------------------------------------------------------------

def mark_value(qc: QuantumCircuit, value: int, n_qubits: int) -> None:
    """Apply a phase flip (-1) to the computational basis state |value>."""
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    # Flip qubits that should be 0 in `value`, so the target pattern becomes
    # all-ones, then apply a multi-controlled Z (via H-MCX-H on last qubit),
    # then undo the flips.
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)


def build_oracle(marked_values, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for v in marked_values:
        mark_value(qc, v, n_qubits)
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


# ---------------------------------------------------------------------------
# 3. Assemble the Grover circuit with the optimal iteration count.
# ---------------------------------------------------------------------------

M = len(classical_primes)  # number of marked states
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Marked count M = {M}, N = {N}, Grover iterations = {iterations}")

oracle = build_oracle(classical_primes, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 20000
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: classical bit c[0] is the rightmost character.
# Convert each measured bitstring back to the integer n it represents.
value_counts = {}
for bitstring, count in counts.items():
    # Qiskit prints classical bits as c[n-1] c[n-2] ... c[0] (leftmost =
    # highest index). c[i] was measured from qubit i, and mark_value treats
    # qubit i as bit i (LSB-first), so reconstruct n_val bit by bit.
    n_val = 0
    for i in range(N_QUBITS):
        bit = int(bitstring[N_QUBITS - 1 - i])
        n_val |= bit << i
    value_counts[n_val] = value_counts.get(n_val, 0) + count

# Take the top-M most frequently measured values as the quantum-found set.
sorted_vals = sorted(value_counts.items(), key=lambda kv: -kv[1])
quantum_top = sorted(v for v, _ in sorted_vals[:M])

print(f"Quantum measurement outcome counts (top {M}): {sorted_vals[:M]}")
print(f"Quantum-found set (top {M} by frequency): {quantum_top}")

# Sanity: the marked states should collectively carry the large majority of
# probability mass after amplitude amplification.
marked_mass = sum(c for v, c in value_counts.items() if v in classical_primes)
print(f"Fraction of shots landing on a marked (prime) state: {marked_mass / shots:.3f}")

passed = (quantum_top == classical_primes) and (marked_mass / shots > 0.5)

if passed:
    print("PASS")
else:
    print("FAIL")
