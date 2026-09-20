"""
Erdos problem #693 (erdosproblems.com) -- quantum-testable instance.

OEIS id used: A391118.

A391118 definition (from OEIS): for n >= 3, let B_n be the set of integers
in [n, n^2] that have a divisor strictly between n and 2n. A391118(n) is
the maximum gap between consecutive elements of B_n. Erdos's conjecture
(problem #693) is that A391118(n) <= (log n)^O(1), i.e. these gaps grow
only polylogarithmically.

Classical property tested here (for the small instance n = 4):

    B_4 = { x in [0, 15] : x has a divisor d with 4 < d < 8 }
        = { x in [0, 15] : x is divisible by 5, 6, or 7 }

restricted to a 4-qubit domain x in {0, ..., 15} (this covers [n, n^2] =
[4, 16] except for the single endpoint x = 16, which cannot change the
maximum gap between consecutive members below it, so A391118(4) computed
over [4,16] equals the same value computed over the restricted domain
[0,15]).

This script:
  1. Computes B_4 and the maximum gap max-gap(B_4) purely classically,
     from first principles (trial division), as ground truth.
  2. Builds a genuine Grover search circuit over 4 qubits whose oracle
     marks exactly the elements of B_4 (via multi-controlled Z gates on
     each marked basis state), with the standard number-of-marked-items
     Grover iteration count, and runs it on the ideal AerSimulator.
  3. Verifies that Grover amplification concentrates measurement
     probability on the true marked set B_4 (quantum search recovers the
     same set membership decided classically), and reports PASS/FAIL.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth
# ---------------------------------------------------------------------------

N_PARAM = 4          # the "n" in A391118(n)
DOMAIN_SIZE = 16      # 4 qubits -> x in [0, 15]
NUM_QUBITS = 4


def has_divisor_in_open_interval(x: int, lo: int, hi: int) -> bool:
    """True if x has a divisor d with lo < d < hi (both integer bounds)."""
    if x == 0:
        return False
    for d in range(lo + 1, hi):
        if d > 0 and x % d == 0:
            return True
    return False


def classical_B_n(n: int, domain_size: int):
    """B_n restricted to the domain [0, domain_size - 1]."""
    lo, hi = n, 2 * n
    return sorted(
        x for x in range(domain_size) if has_divisor_in_open_interval(x, lo, hi)
    )


def max_gap(sorted_members):
    if len(sorted_members) < 2:
        return 0
    return max(b - a for a, b in zip(sorted_members, sorted_members[1:]))


B4 = classical_B_n(N_PARAM, DOMAIN_SIZE)
A391118_AT_4 = max_gap(B4)

print(f"Classical B_4 (restricted to [0,15]) = {B4}")
print(f"Classical max gap (A391118(4), restricted domain) = {A391118_AT_4}")

# Sanity check against the full [n, n^2] = [4, 16] domain: element 16 is
# divisible by 5,6,7? no (16 % 5, %6, %7 all nonzero), so it is not a
# member and adding it back cannot change the maximum gap below it.
assert not has_divisor_in_open_interval(16, N_PARAM, 2 * N_PARAM)
assert B4 == [5, 6, 7, 10, 12, 14, 15]
assert A391118_AT_4 == 3


# ---------------------------------------------------------------------------
# 2. Grover search circuit marking exactly the elements of B_4
# ---------------------------------------------------------------------------

def bitstring(x: int, num_qubits: int) -> str:
    return format(x, f"0{num_qubits}b")


def apply_marking_multi_controlled_z(qc: QuantumCircuit, qubits, x: int, num_qubits: int):
    """Flip the phase of basis state |x> using a multi-controlled Z."""
    bits = bitstring(x, num_qubits)  # MSB first, matches qubits[0]=MSB convention below
    # Open-control on 0-bits: X before and after the controlled-Z.
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    for i in zero_positions:
        qc.x(qubits[i])
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for i in zero_positions:
        qc.x(qubits[i])


def build_oracle(marked, num_qubits):
    qc = QuantumCircuit(num_qubits, name="Oracle")
    for x in marked:
        apply_marking_multi_controlled_z(qc, list(range(num_qubits)), x, num_qubits)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def grover_iterations(num_marked: int, domain_size: int) -> int:
    theta = math.asin(math.sqrt(num_marked / domain_size))
    r = round((math.pi / (4 * theta)) - 0.5)
    return max(1, r)


marked_set = B4
num_marked = len(marked_set)
iterations = grover_iterations(num_marked, DOMAIN_SIZE)

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

oracle = build_oracle(marked_set, NUM_QUBITS)
diffuser = build_diffuser(NUM_QUBITS)

for _ in range(iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

print(f"\nNumber marked = {num_marked}, Grover iterations = {iterations}")


# ---------------------------------------------------------------------------
# 3. Run on ideal AerSimulator and verify
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical bit order in the count keys is little-endian relative
# to qubit index (qubit 0 -> rightmost char). Our oracle/diffuser addressed
# qubits[0] as the MSB of the *integer x* via `bitstring`, so to recover x
# from a measured bitstring we must reverse it before int(..., 2).
def counts_key_to_int(key: str) -> int:
    return int(key[::-1], 2)

outcome_counts = {}
for key, c in counts.items():
    x = counts_key_to_int(key)
    outcome_counts[x] = outcome_counts.get(x, 0) + c

sorted_outcomes = sorted(outcome_counts.items(), key=lambda kv: -kv[1])
print("\nTop measured outcomes (value: count):")
for x, c in sorted_outcomes[:10]:
    tag = "MEMBER of B_4" if x in marked_set else "not a member"
    print(f"  {x:2d} ({c:4d} shots, {100*c/SHOTS:5.1f}%) - {tag}")

marked_shots = sum(c for x, c in outcome_counts.items() if x in marked_set)
marked_fraction = marked_shots / SHOTS

# Also check: the single most probable outcome must itself be a true
# member of B_4, i.e. quantum search actually recovers a genuine element
# of the classically-defined set, not an arbitrary/incorrect value.
most_probable_x, _ = sorted_outcomes[0]
top_hit_is_member = most_probable_x in marked_set

print(f"\nFraction of shots landing on a true member of B_4: {marked_fraction:.3f}")
print(f"Most probable measured value: {most_probable_x} "
      f"({'is' if top_hit_is_member else 'is NOT'} a member of B_4)")

# Grover amplification with 7/16 marked and 1 iteration should push the
# marked-set probability well above the uniform baseline of 7/16 ~ 0.4375.
uniform_baseline = num_marked / DOMAIN_SIZE
verified = top_hit_is_member and marked_fraction > uniform_baseline

print(f"\nUniform baseline (no amplification) = {uniform_baseline:.3f}")
print(f"Amplified fraction                 = {marked_fraction:.3f}")

if verified:
    print("\nPASS: Grover search on the quantum circuit concentrates on true "
          "members of B_4 (Erdos problem #693 / A391118), matching the "
          "classical ground truth computed from first principles "
          f"(A391118(4) = {A391118_AT_4}).")
else:
    print("\nFAIL: quantum result did not match/verify against the "
          "classical ground truth.")

assert verified, "Quantum Grover search result did not verify against classical answer"
