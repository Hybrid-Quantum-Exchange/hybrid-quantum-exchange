"""
Erdos problem #369 -- quantum-testable sequence lane.

Erdos problem #369's entry in data/problems.yaml (manman4/erdosproblems,
`erdosproblems.com` dataset) is a resolved number-theory statement:

    number: "369"
    prize: "no"
    status: proved (Lean), last update 2026-03-27
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (reported honestly, per the task instructions): problem #369 has
no associated OEIS sequence id in the source data ("N/A"). Without an OEIS
id there is no specific integer sequence to derive a finite, checkable
membership/divisibility/counting property FROM for this problem -- there is
nothing problem-369-specific to encode into a quantum oracle. Fabricating a
property and attributing it to problem 369 would misrepresent the source
data, which the task instructions explicitly forbid.

So this script does not verify anything about problem #369 itself.

Instead, as the instructed best-honest-effort fallback, it demonstrates a
genuine, fully worked quantum circuit for the closest legitimate stand-in
available -- an unconditional finite number-theory search of the kind
problem #369's own tag ("number theory") describes: Grover's algorithm
searching the 4-bit space {0, 1, ..., 15} (N = 16, 4 qubits) for the
integers that are PRIME. This is a real, checkable, finite, computable
property (primality of small integers), tested with a real quantum search
circuit against a first-principles classical computation of the same
property, but it is a generic demonstration and NOT a fact about, or
derived from, Erdos problem #369 or any OEIS sequence tied to it.

Classical ground truth (computed here, trial division, not looked up):
    primes in [0, 15] = {2, 3, 5, 7, 11, 13}   (6 of the 16 values)

Quantum approach:
    - 4 qubits encode n in [0, 15] in binary (q0 = LSB .. q3 = MSB).
    - A phase oracle flips the sign of the amplitude of every basis state
      whose integer value is prime, implemented directly from the
      classical `is_prime` truth table via a multi-controlled-Z per marked
      state (a bona fide oracle construction, not a shortcut).
    - The Grover diffuser (inversion about the mean) is applied for the
      near-optimal number of iterations for M=6 marked items out of N=16.
    - The circuit is run on the ideal AerSimulator (statevector-exact via
      sampling with many shots) and the most frequently measured outcomes
      are compared against the classical prime set.

PASS criterion: every one of the classical primes in [0,15] appears among
the quantum circuit's top-6 most-measured outcomes (Grover search
amplifies exactly the marked states, so with M=6 marked items among N=16
the 6 highest-probability outcomes should be exactly the primes).
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles (trial division).
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
M = len(classical_primes)
print(f"Classical primes in [0, {N - 1}]: {classical_primes}  (M = {M})")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classical truth table directly.
# ---------------------------------------------------------------------------
def apply_multi_controlled_z_for_value(qc: QuantumCircuit, value: int, n_qubits: int) -> None:
    """Flip the phase of the |value> basis state only, using X-gates to map
    the target bit pattern onto all-ones, a multi-controlled Z, then undo
    the X-gates."""
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    flip_qubits = [i for i, b in enumerate(bits) if b == 0]

    for q in flip_qubits:
        qc.x(q)

    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    for q in flip_qubits:
        qc.x(q)


def build_oracle(marked_values, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for v in marked_values:
        apply_multi_controlled_z_for_value(qc, v, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(classical_primes, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

# Near-optimal number of Grover iterations for M marked items out of N.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------
backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 20000
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order: rightmost classical bit is qubit 0 (LSB) already,
# so int(bitstring, 2) directly gives n.
value_counts = {}
for bitstring, count in counts.items():
    n = int(bitstring, 2)
    value_counts[n] = value_counts.get(n, 0) + count

ranked = sorted(value_counts.items(), key=lambda kv: -kv[1])
top_M = sorted(v for v, _ in ranked[:M])

print("Measurement counts by integer value:")
for v in range(N):
    print(f"  n={v:2d}  count={value_counts.get(v, 0):5d}  classical_prime={is_prime(v)}")
print(f"Top-{M} most measured values (quantum): {top_M}")
print(f"Classical primes                       : {classical_primes}")


# ---------------------------------------------------------------------------
# 4. Compare quantum result against the classical answer.
# ---------------------------------------------------------------------------
verified = (top_M == classical_primes)

if verified:
    print("PASS")
else:
    print("FAIL")
