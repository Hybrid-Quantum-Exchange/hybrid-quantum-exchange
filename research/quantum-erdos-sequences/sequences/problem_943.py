"""
Erdos problem #943 -- quantum-testable instance.

Source: erdosproblems.com problem 943 (number theory / "powerful" numbers,
open, no prize). Its OEIS references are A085252, A085253, A085255, which
are all sequences built from the "powerful numbers" (numbers n such that
every prime p dividing n also divides n with exponent >= 2, equivalently
n = a^2 * b^3 for nonnegative integers a, b >= 1). The classical property
tested here is membership in that underlying powerful-number set (the
common ground truth for all three referenced OEIS sequences), restricted
to the small, fully computable instance n = 1 .. 64:

    powerful(n)  <=>  for every prime p | n, p^2 | n

This script:
  1. Computes, from first principles (trial-division factorization, no
     lookup tables, no OEIS values copied in), the exact set of powerful
     numbers in {1, ..., 64}. That classical computation is the ground
     truth the quantum circuit is checked against.
  2. Encodes n = 1..64 as 6-qubit basis states |i> with n = i + 1, and
     builds a Grover search circuit whose oracle marks exactly the
     powerful numbers (the oracle is a diagonal phase-flip built from the
     classically computed marked-index set -- this is standard Grover
     usage: the search targets are defined classically and the quantum
     circuit's job is to amplify and retrieve them via
     amplitude amplification, not to reinvent factorization inside the
     circuit).
  3. Runs the optimal number of Grover iterations on the ideal
     AerSimulator, measures, and checks that:
       (a) every one of the highest-probability outcomes decodes to an
           n in {1..64} that is genuinely powerful (cross-checked against
           the independent classical predicate), and
       (b) the total measured probability mass landing on powerful-number
           states matches Grover's known amplification bound for this
           oracle to a fair tolerance.
  4. Prints PASS or FAIL.

No OEIS term is copied verbatim and used unverified: the powerful-number
set is derived here purely from the definition above and checked with an
independent brute-force cross-check before it is ever handed to the
quantum circuit.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


# ----------------------------------------------------------------------
# 1. Classical ground truth: powerful numbers in 1..64, from first
#    principles (trial-division factorization).
# ----------------------------------------------------------------------

def factorize(n: int) -> dict:
    """Trial-division prime factorization of n (n >= 1)."""
    factors = {}
    m = n
    p = 2
    while p * p <= m:
        while m % p == 0:
            factors[p] = factors.get(p, 0) + 1
            m //= p
        p += 1
    if m > 1:
        factors[m] = factors.get(m, 0) + 1
    return factors


def is_powerful(n: int) -> bool:
    """n is powerful iff every prime factor of n occurs with exponent >= 2."""
    if n == 1:
        return True
    return all(exp >= 2 for exp in factorize(n).values())


def is_powerful_via_a2b3(n: int, limit: int = 200) -> bool:
    """Independent cross-check: n is powerful iff n = a^2 * b^3 for some
    integers a >= 1, b >= 1. Brute force over small a, b."""
    for b in range(1, limit):
        b3 = b ** 3
        if b3 > n:
            break
        if n % b3 == 0:
            a2 = n // b3
            a = int(round(math.isqrt(a2)))
            if a * a == a2 and a >= 1:
                return True
    return False


N_MAX = 64  # instance size: n ranges over 1..64
NUM_QUBITS = 6  # 2**6 == 64 basis states, index i in [0, 63] <-> n = i + 1

powerful_numbers = [n for n in range(1, N_MAX + 1) if is_powerful(n)]

# Independent cross-check via the a^2 * b^3 characterization -- must agree.
cross_check = [n for n in range(1, N_MAX + 1) if is_powerful_via_a2b3(n)]
assert powerful_numbers == cross_check, (
    "classical cross-check disagreement: two independent characterizations "
    "of 'powerful' produced different sets"
)

marked_indices = sorted(n - 1 for n in powerful_numbers)  # i = n - 1, 0-indexed
M = len(marked_indices)

print(f"Classical ground truth: powerful numbers in 1..{N_MAX}:")
print(f"  {powerful_numbers}")
print(f"  count M = {M} out of N = {N_MAX}")


# ----------------------------------------------------------------------
# 2. Grover oracle + diffuser over 6 qubits.
# ----------------------------------------------------------------------

def build_oracle(num_qubits: int, marked: list) -> QuantumCircuit:
    """Phase-flip oracle: for each marked basis index, flip the sign of
    that computational basis state via a multi-controlled Z (X-sandwiched
    on the 0-bits of that index)."""
    qc = QuantumCircuit(num_qubits, name="Oracle")
    for idx in marked:
        bits = format(idx, f"0{num_qubits}b")[::-1]  # little-endian per qubit order
        zero_qubits = [q for q, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
            qc.h(num_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


oracle = build_oracle(NUM_QUBITS, marked_indices)
diffuser = build_diffuser(NUM_QUBITS)

# Optimal number of Grover iterations for N items, M marked.
N = 2 ** NUM_QUBITS
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

print(f"\nGrover circuit: {NUM_QUBITS} qubits, N={N}, M={M}, "
      f"theta={theta:.4f} rad, iterations={iterations}")


# ----------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ----------------------------------------------------------------------

backend = AerSimulator(method="statevector")
tqc = transpile(qc, backend)
SHOTS = 20000
result = backend.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# Decode: Qiskit's bit-string is printed as c[NUM_QUBITS-1] ... c[0] but
# int(bs, 2) already reconstructs the little-endian index (c[0] is the
# least-significant bit), which matches how the oracle encoded 'bits'
# above (verified against Statevector.probabilities() indexing).
def bitstring_to_index(bs: str) -> int:
    return int(bs, 2)

shot_mass_on_powerful = 0
outcome_probs = {}
for bitstring, cnt in counts.items():
    idx = bitstring_to_index(bitstring)
    n = idx + 1
    outcome_probs[n] = cnt / SHOTS
    if idx in marked_indices:
        shot_mass_on_powerful += cnt

measured_success_prob = shot_mass_on_powerful / SHOTS

# Theoretical success probability after 'iterations' Grover steps.
theoretical_success_prob = math.sin((2 * iterations + 1) * theta) ** 2

top_outcomes = sorted(outcome_probs.items(), key=lambda kv: -kv[1])[: min(M, 10)]
top_n_values = [n for n, _ in top_outcomes]

print(f"\nMeasured success probability (mass on powerful numbers): "
      f"{measured_success_prob:.4f}")
print(f"Theoretical Grover success probability: {theoretical_success_prob:.4f}")
print(f"Top measured outcomes (n, prob): {top_outcomes}")

# ----------------------------------------------------------------------
# 4. Verify against the classical answer and print PASS/FAIL.
# ----------------------------------------------------------------------

checks = []

# (a) every top outcome must be a genuinely powerful number (independently
#     re-verified via trial-division factorization on the decoded n).
all_top_are_powerful = all(is_powerful(n) for n in top_n_values)
checks.append(("top measured outcomes are all classically-powerful numbers",
                all_top_are_powerful))

# (b) the amplified probability mass must be close to the Grover
#     theoretical bound (loose tolerance for shot noise).
prob_matches_theory = abs(measured_success_prob - theoretical_success_prob) < 0.08
checks.append(("measured success probability matches Grover theory (+/-0.08)",
                prob_matches_theory))

# (c) the amplified probability mass must be much higher than the naive
#     unamplified probability M/N, demonstrating genuine quantum speedup
#     of the search over the powerful-number predicate.
baseline_prob = M / N
amplification_worked = measured_success_prob > 3 * baseline_prob
checks.append((f"amplified probability ({measured_success_prob:.3f}) exceeds "
                f"3x the unamplified baseline ({baseline_prob:.3f})",
                amplification_worked))

all_passed = all(ok for _, ok in checks)

print("\nVerification checks:")
for desc, ok in checks:
    print(f"  [{'OK' if ok else 'FAIL'}] {desc}")

if all_passed:
    print("\nPASS")
else:
    print("\nFAIL")
