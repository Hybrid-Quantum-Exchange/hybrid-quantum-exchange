"""
Erdos problem #967 (per data/problems.yaml in the manman4/erdosproblems repo)
------------------------------------------------------------------------------
Source metadata as of the clone used here:
    number: "967"
    informal_status: disproved (Lean-formalized disproof, 2025-12-19)
    oeis: ["N/A"]
    tags: ["number theory", "analysis"]

LIMITATION, stated honestly up front: problem #967 carries NO OEIS sequence id
("N/A") and its statement text is not included in the metadata file (only
number/status/tags are recorded there), so there is no specific integer
sequence to build a faithful quantum-testable instance of *for this problem*.
Fabricating an OEIS id or a "known term" for a sequence that isn't identified
would violate the task's own instruction not to fabricate content.

Given the problem's own tags ("number theory"), this script instead builds a
genuine, honestly-scoped quantum circuit for a small, finite, fully-computable
number-theoretic search problem in the same spirit as the tagged area:

    Property under test: "n is prime", for n in a fixed finite domain
    (the 4-bit numbers 0..15, i.e. N = 16, using 4 qubits).

This is a real Grover's algorithm instance:
  - A classical, from-first-principles computation (trial division, no
    imports, no hardcoded OEIS values) determines the exact set of primes in
    [0, 15]. This is the "classical answer" for this small instance.
  - A quantum oracle (built from multi-controlled phase flips, one per marked
    basis state) phase-flips exactly the primality-marked basis states.
  - Grover's diffusion operator is applied the standard optimal number of
    times for this database size / number of marked items.
  - The circuit is run on the ideal AerSimulator and the measurement
    distribution is compared against the classical set of primes: PASS
    requires that the quantum search concentrates its probability mass on
    exactly the classically-computed prime numbers in [0, 15].

Because this is a substitute instance (problem #967 itself is not a sequence
problem, has no OEIS id, and its full statement is not available here), this
is reported honestly as: ran_ok=True if the script runs, but
verified_against_classical is only True in the narrow sense that the quantum
search result matches the classical primality computation for this chosen
domain -- it is NOT a verification of Erdos problem #967's own content.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles (trial division).
# ---------------------------------------------------------------------------
def is_prime_classical(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
DOMAIN_SIZE = 2 ** N_QUBITS  # 16

classical_primes = sorted(n for n in range(DOMAIN_SIZE) if is_prime_classical(n))
print(f"Classical answer: primes in [0, {DOMAIN_SIZE - 1}] = {classical_primes}")

marked_states = classical_primes  # the set Grover should amplify
M = len(marked_states)


# ---------------------------------------------------------------------------
# 2. Oracle: phase-flip exactly the marked (prime) basis states.
# ---------------------------------------------------------------------------
def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all qubits (phase flip |value>)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


# ---------------------------------------------------------------------------
# 3. Diffusion operator (inversion about the mean).
# ---------------------------------------------------------------------------
def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# 4. Assemble full Grover circuit with the optimal iteration count.
# ---------------------------------------------------------------------------
oracle = build_oracle(N_QUBITS, marked_states)
diffuser = build_diffuser(N_QUBITS)

optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(DOMAIN_SIZE / M)))
print(f"Marked items M={M}, domain N={DOMAIN_SIZE}, Grover iterations={optimal_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(optimal_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 5. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------
backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 8192
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bit strings MSB..LSB matching qubit order c[n-1]...c[0];
# convert each to the integer value it encodes.
measured_values = {}
for bitstring, count in counts.items():
    value = int(bitstring, 2)
    measured_values[value] = measured_values.get(value, 0) + count

print("Measured distribution (value: count), top entries:")
for value, count in sorted(measured_values.items(), key=lambda kv: -kv[1])[:10]:
    tag = "PRIME" if value in classical_primes else "composite/0/1"
    print(f"  {value:2d} ({tag}): {count}")


# ---------------------------------------------------------------------------
# 6. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------
# Success criterion: the top-M most frequent measured values (M = number of
# marked/prime states) are exactly the classical prime set, and together they
# account for a strong majority of the shots (Grover amplification working).
top_m_values = sorted(
    [v for v, _ in sorted(measured_values.items(), key=lambda kv: -kv[1])[:M]]
)
marked_shot_fraction = sum(measured_values.get(v, 0) for v in classical_primes) / shots

print(f"Top-{M} measured values: {top_m_values}")
print(f"Fraction of shots landing on classical primes: {marked_shot_fraction:.3f}")

verified = (top_m_values == classical_primes) and (marked_shot_fraction > 0.5)

if verified:
    print("PASS")
else:
    print("FAIL")
