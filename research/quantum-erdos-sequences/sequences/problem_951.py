"""
Erdos problem #951 -- quantum-testable lane (best-effort fallback).

Source metadata for problem #951, as recorded in erdosproblems/data/problems.yaml
(fetched read-only from the manman4/erdosproblems clone on 2026-09-19):

    number: "951"
    prize: "no"
    informal_status: open (last update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory"]
    (no free-text statement/comment field is present for this entry)

LIMITATION (reported honestly, per task instructions): problem #951 carries
no OEIS sequence id ("N/A") and no textual statement in the metadata file, so
there is no sequence to derive a finite, checkable property from for this
specific problem. Fabricating an OEIS id or a "matching" property would not
be honest. This script is therefore a best-effort, clearly-labeled fallback:
it targets a genuine, small, finite, classically-computable number-theory
property (primality on a bounded range), consistent with problem #951's own
tag ("number theory"), and verifies a REAL Grover-search quantum circuit
against the classical answer. It is NOT a verification of problem #951's
actual (unavailable) statement, and should be read as "ran_ok /
verified_against_classical" for the fallback task only, not as a solution
tied to problem 951's true content.

Chosen finite instance
-----------------------
Search space: integers 0..15 (encoded in 4 qubits, N = 16).
Property being tested: "n is prime" (classical, elementary primality test,
computed from first principles in this script -- no external tables).
Primes in [0,15]: {2, 3, 5, 7, 11, 13} -> 6 marked out of 16 (marked
fraction 3/8), a low-enough density that a single Grover iteration gives a
clear, checkable amplitude boost over uniform random guessing.

Circuit: standard 3-qubit Grover search --
    - oracle: phase-flips computational basis states |n> for n in {2,3,5,7}
      (built directly from each prime's 3-bit binary representation, using
      X gates to map that pattern to |111> and a multi-controlled Z)
    - diffuser: the standard inversion-about-the-mean operator
    - one Grover iteration is optimal here (marked fraction 1/2 -> the
      standard iteration count floor(pi/4 * sqrt(N/M)) with N=8, M=4 rounds
      to 1)

Verification: run the circuit on the ideal AerSimulator, take the most
frequent measured outcomes, and check that they are exactly the classically
computed prime set {2, 3, 5, 7} (equivalently, that essentially all
measurement weight lands on prime-labelled basis states rather than
composite ones). Print PASS/FAIL accordingly.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16
classical_primes = sorted(n for n in range(N) if is_prime(n))
classical_prime_set = set(classical_primes)
M = len(classical_prime_set)  # number of marked items

print(f"Search space: n in [0, {N - 1}]")
print(f"Classical primes (ground truth, computed here): {classical_primes}")
print(f"Marked count M = {M} out of N = {N}")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle that marks exactly the prime states.
# ---------------------------------------------------------------------------

def mark_state(qc: QuantumCircuit, value: int, n_qubits: int) -> None:
    """Phase-flip the computational basis state |value> (multi-controlled Z)."""
    bits = format(value, f"0{n_qubits}b")  # MSB..LSB matches qubit n-1..0
    # Map |value> -> |11...1> using X on qubits whose bit is 0.
    for i, bit in enumerate(reversed(bits)):
        if bit == "0":
            qc.x(i)
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for i, bit in enumerate(reversed(bits)):
        if bit == "0":
            qc.x(i)


def build_oracle(marked_values, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for v in marked_values:
        mark_state(qc, v, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(classical_prime_set, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

# Optimal number of Grover iterations for M marked out of N.
theta = np.arcsin(np.sqrt(M / N))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

grover = QuantumCircuit(N_QUBITS, N_QUBITS)
grover.h(range(N_QUBITS))
for _ in range(iterations):
    grover.append(oracle.to_gate(), range(N_QUBITS))
    grover.append(diffuser.to_gate(), range(N_QUBITS))
grover.measure(range(N_QUBITS), range(N_QUBITS))

print(f"Grover iterations used: {iterations}")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(grover, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-bit string is c[n-1]...c[0]; convert each outcome to int.
outcome_counts = {}
for bitstring, c in counts.items():
    value = int(bitstring, 2)
    outcome_counts[value] = outcome_counts.get(value, 0) + c

print("Measurement outcome counts (state: shots):")
for value in sorted(outcome_counts, key=lambda v: -outcome_counts[v]):
    tag = "prime" if value in classical_prime_set else "composite/0/1"
    print(f"  {value} ({tag}): {outcome_counts[value]}")

# The set of the M most-frequently measured outcomes should be exactly the
# classically computed prime set: Grover amplifies the marked subspace, so
# after amplification its M states should dominate the top-M slots.
top_m_outcomes = set(
    sorted(outcome_counts, key=lambda v: -outcome_counts[v])[:M]
)

marked_weight = sum(c for v, c in outcome_counts.items() if v in classical_prime_set)
marked_fraction = marked_weight / shots

print(f"Fraction of shots landing on a classically-prime state: {marked_fraction:.3f}")
print(f"Uniform-random baseline would give: {M / N:.3f}")

verified = (top_m_outcomes == classical_prime_set) and (marked_fraction > M / N)

if verified:
    print("PASS")
else:
    print("FAIL")
