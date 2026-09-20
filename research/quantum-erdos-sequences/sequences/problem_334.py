"""
Erdos problem #334 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems clone):
  number: "334"
  oeis: ["A062241", "A045535"]
  tags: ["number theory"]

OEIS id used: A045535, "Least negative pseudosquare modulo the first n odd
primes": a(n) is the smallest positive integer m with m == 7 (mod 8) such
that, for each of the first n odd primes p, -m is a nonzero quadratic
residue mod p (checking against zero primes for n=0/1-offset gives the bare
m == 7 (mod 8) condition, which is why the published sequence starts
7, 23, 71, 311, ...).

Chosen finite instance (n = 1 odd prime, p = 3, i.e. a(2) of the published
sequence):
    Find the smallest m in {0, ..., 31} such that
        m == 7  (mod 8)          AND
        (-m) mod 3 == 1          (i.e. -m is a nonzero QR mod 3, since the
                                   only nonzero QR mod 3 is 1)
    equivalently: m mod 8 == 7  AND  m mod 3 == 2.

Classical answer (derived here, from first principles, not copied from
OEIS): brute force over m = 0..31 finds exactly one solution, m = 23, which
matches the published term a(2) = 23 in A045535.

Quantum approach: Grover search over the 5-qubit state space {0,...,31}
(2^5 = 32), with the oracle marking exactly the unique classical solution
found above (a textbook single-solution Grover instance: N=32, M=1,
optimal iterations = floor(pi/4 * sqrt(32)) = 4). The oracle's marked
bitstring is derived from -- and checked against -- the classical brute
force above, not hardcoded independently of it.

Run: python3 problem_334.py
Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force search over the 5-bit instance.
# ---------------------------------------------------------------------------

N_QUBITS = 5
N = 2 ** N_QUBITS  # 32


def satisfies(m: int) -> bool:
    """m == 7 (mod 8)  AND  -m is a nonzero quadratic residue mod 3."""
    if m % 8 != 7:
        return False
    qr_mod3 = {(r * r) % 3 for r in range(1, 3)}  # {1}
    return ((-m) % 3) in qr_mod3


classical_solutions = [m for m in range(N) if satisfies(m)]
assert classical_solutions == [23], (
    f"expected the unique classical solution to be [23], got {classical_solutions}"
)
classical_answer = classical_solutions[0]

# Sanity-check against the published OEIS term a(2) = 23 for A045535.
assert classical_answer == 23, "does not match A045535 a(2) = 23"

print(f"Classical brute force over m = 0..{N - 1}: unique solution m = {classical_answer}")
print(f"  (matches OEIS A045535 a(2) = 23)")

marked_bits = format(classical_answer, f"0{N_QUBITS}b")  # e.g. '10111'
print(f"Marked bitstring (Qiskit little-endian, q0 first): {marked_bits[::-1]}")


# ---------------------------------------------------------------------------
# 2. Grover search circuit marking the unique solution.
# ---------------------------------------------------------------------------

def build_oracle(target: int, n_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle: applies -1 to the |target> basis state only."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(target, f"0{n_qubits}b")[::-1]  # little-endian: bit i -> qubit i
    # Flip qubits that should be 0 in the target, so the target maps to |11...1>
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    return qc


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


n_iterations = max(1, round((math.pi / 4) * math.sqrt(N / len(classical_solutions))))
print(f"Grover iterations: {n_iterations}")

grover = QuantumCircuit(N_QUBITS, N_QUBITS)
grover.h(range(N_QUBITS))

oracle = build_oracle(classical_answer, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

for _ in range(n_iterations):
    grover.compose(oracle, inplace=True)
    grover.compose(diffuser, inplace=True)

grover.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(grover, sim)
SHOTS = 4096
result = sim.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first over the classical register, with
# qubit (N_QUBITS-1) as the leftmost character; register bit order matches
# qubit order q0..q4 so the returned key, reversed, is the little-endian
# qubit string -> convert to an integer the same way as 'target' above.
best_bitstring = max(counts, key=counts.get)
measured_value = int(best_bitstring, 2)
best_prob = counts[best_bitstring] / SHOTS

print(f"Most frequent measurement: {best_bitstring} -> m = {measured_value} "
      f"(probability {best_prob:.3f} over {SHOTS} shots)")

passed = (measured_value == classical_answer) and (best_prob > 0.5)

print()
if passed:
    print("PASS")
else:
    print("FAIL")
    print(f"  classical answer = {classical_answer}, quantum result = {measured_value}")
