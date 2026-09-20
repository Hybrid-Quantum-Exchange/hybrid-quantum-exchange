"""
Erdos problem #83 -- quantum-testable instance.

Erdos problem #83 (see erdosproblems.com/83, tags: combinatorics; prize $500,
status "proved (Lean)" as of 2026-08-24) is linked in the problems dataset to
OEIS sequences A071799 and A387635. This script uses A071799.

A071799: "Number of lattice paths in the lattice [0..2n] X [0..2n] which do
not pass through the point (n,n)."  OEIS gives the closed form

    a(n) = C(4n, 2n) - C(2n, n)^2

and the listed terms 2, 34, 524, 7970, ... starting at n=1.

Classical property tested here (computed from first principles, not copied):
For n = 1, a monotone lattice path from (0,0) to (2,2) using unit steps Right
(R) and Up (U) can be encoded as a length-4 bit string over {R=0, U=1} with
exactly two 1-bits (there are C(4,2) = 6 such strings/paths in total). The
path visits the center point (1,1) exactly when its first two steps contain
exactly one R and one U (i.e. after 2 steps it has taken one step in each
direction). So the paths that AVOID (1,1) are exactly those length-4,
weight-2 bit strings whose first two bits are EQUAL (00 or 11); completing
each with the remaining required 1-bits gives exactly two strings:

    0011  and  1100

which matches a(1) = 2 from the OEIS formula: C(4,2) - C(2,1)^2 = 6 - 4 = 2.

This script:
  1. Computes a(1) classically two ways (direct enumeration of the 6 length-4
     weight-2 bit strings, and the closed-form binomial formula) and checks
     they agree.
  2. Builds a genuine Grover search circuit over the 4-qubit space of all
     16 length-4 bit strings, whose oracle marks exactly the states that
     (a) have Hamming weight 2 (a valid path from (0,0) to (2,2)) AND
     (b) avoid the center point (1,1), i.e. are one of {0011, 1100}.
  3. Runs the circuit on the ideal AerSimulator and checks that Grover's
     algorithm amplifies exactly the two classically-predicted marked
     strings, matching them against the classical answer above.
  4. Prints PASS/FAIL.

Only qiskit, qiskit_aer and numpy are used.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of a(1), from first principles.
# ---------------------------------------------------------------------------

def classical_marked_strings(n_bits: int = 4, weight: int = 2):
    """Enumerate all length-n_bits bit strings of the given Hamming weight
    that encode a lattice path from (0,0) to (2,2) avoiding the center
    point (1,1). Bit i is 1 for an Up step, 0 for a Right step."""
    marked = []
    for bits in itertools.product([0, 1], repeat=n_bits):
        if sum(bits) != weight:
            continue  # not a valid path to (2,2)
        # position after the first 2 steps:
        ups_so_far = sum(bits[:2])
        rights_so_far = 2 - ups_so_far
        # center point for n=1 is (1,1): reached iff exactly one R and one U
        # among the first two steps.
        passes_through_center = (ups_so_far == 1 and rights_so_far == 1)
        if not passes_through_center:
            marked.append("".join(str(b) for b in bits))
    return sorted(marked)


def binomial(a, b):
    return math.comb(a, b)


def a_n_formula(n: int) -> int:
    """a(n) = C(4n, 2n) - C(2n, n)^2, per OEIS A071799."""
    return binomial(4 * n, 2 * n) - binomial(2 * n, n) ** 2


classical_marked = classical_marked_strings()
a1_enumeration = len(classical_marked)
a1_formula = a_n_formula(1)

print("Classical check (n = 1):")
print(f"  Marked bit strings (paths avoiding center (1,1)): {classical_marked}")
print(f"  Count by direct enumeration: {a1_enumeration}")
print(f"  Count by OEIS A071799 formula C(4,2) - C(2,1)^2: {a1_formula}")

assert a1_enumeration == a1_formula == 2, "classical cross-check failed"
assert classical_marked == ["0011", "1100"], "unexpected marked set"

MARKED_STATES = classical_marked  # e.g. ["0011", "1100"]
N_QUBITS = 4
N_TOTAL = 2 ** N_QUBITS


# ---------------------------------------------------------------------------
# 2. Grover search circuit marking exactly MARKED_STATES among 16 strings.
# ---------------------------------------------------------------------------

def apply_multi_controlled_z_on_pattern(qc: QuantumCircuit, qubits, bitstring: str):
    """Flip the phase of the single computational basis state matching
    `bitstring` (qiskit little-endian: bitstring[0] -> qubits[0], the
    convention used consistently throughout this script), leaving all
    other basis states untouched. Standard "mark one computational basis
    state" building block: X on the 0-bits, multi-controlled Z, X again.
    """
    zero_positions = [i for i, b in enumerate(bitstring) if b == "0"]
    for i in zero_positions:
        qc.x(qubits[i])

    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])

    for i in zero_positions:
        qc.x(qubits[i])


def oracle(qc: QuantumCircuit, qubits, marked_states):
    for s in marked_states:
        apply_multi_controlled_z_on_pattern(qc, qubits, s)


def diffuser(qc: QuantumCircuit, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(marked_states, n_qubits, n_total):
    m = len(marked_states)
    # Standard optimal Grover iteration count for m marked out of n_total.
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_total / m)))

    qreg = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(qreg)
    qc.h(qreg)  # uniform superposition over all 16 bit strings

    for _ in range(iterations):
        oracle(qc, qreg, marked_states)
        diffuser(qc, qreg)

    qc.measure_all()
    return qc, iterations


qc, iterations = build_grover_circuit(MARKED_STATES, N_QUBITS, N_TOTAL)

print(f"\nGrover circuit: {N_QUBITS} qubits, {len(MARKED_STATES)} marked states, "
      f"{iterations} Grover iteration(s).")
print(qc.draw(output="text"))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
transpiled = transpile(qc, backend)
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# qiskit's measure_all() produces bitstrings in Qiskit's little-endian
# convention: counts key order is q_{n-1}...q_0. We built the state so that
# qreg[0] is the leftmost character of our "bitstring" (as fed to
# apply_multi_controlled_z_on_pattern), so reverse the measured key to
# compare against our bitstring convention.
def to_our_convention(qiskit_bitstring: str) -> str:
    return qiskit_bitstring[::-1]

converted_counts = {}
for bitstr, c in counts.items():
    converted_counts[to_our_convention(bitstr)] = converted_counts.get(to_our_convention(bitstr), 0) + c

sorted_counts = sorted(converted_counts.items(), key=lambda kv: -kv[1])
print("\nMeasurement outcomes (top 6, our bit convention, qubit0=leftmost):")
for bitstr, c in sorted_counts[:6]:
    tag = " <-- marked" if bitstr in MARKED_STATES else ""
    print(f"  {bitstr}: {c}/{shots}{tag}")

marked_shots = sum(converted_counts.get(s, 0) for s in MARKED_STATES)
marked_fraction = marked_shots / shots

print(f"\nFraction of shots landing on a classically-marked state: "
      f"{marked_fraction:.4f} ({marked_shots}/{shots})")

# The two most frequent outcomes should be exactly the two marked states,
# and together they should dominate the distribution (Grover amplification).
top2 = {bitstr for bitstr, _ in sorted_counts[:2]}
quantum_matches_classical = (top2 == set(MARKED_STATES)) and (marked_fraction > 0.85)


# ---------------------------------------------------------------------------
# 4. Verdict.
# ---------------------------------------------------------------------------

print(f"\nClassical answer (paths avoiding center, n=1): {sorted(MARKED_STATES)}")
print(f"Quantum result (top measured states):           {sorted(top2)}")

if quantum_matches_classical:
    print("\nPASS")
else:
    print("\nFAIL")
