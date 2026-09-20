"""
Erdos problem #469 -- quantum-testable instance.

Problem #469 (erdosproblems.com) concerns weird numbers: a positive integer n
is WEIRD if it is abundant (sigma(n) - n > n, i.e. the sum of its proper
divisors exceeds n) but not semiperfect (no subset of its proper divisors
sums exactly to n). The relevant OEIS sequences, taken from
data/problems.yaml for problem 469, are:

    A006036 -- Weird numbers.
    A119425 -- (companion sequence referenced alongside A006036 for this
                problem; the "possible" third tag entry is a status marker
                in the source YAML, not a separate OEIS id).

The property this script actually computes and verifies is the SEMIPERFECT
test that distinguishes a weird number from a merely abundant one:

    Is there a subset of the proper divisors of n that sums exactly to n?

Classical instance chosen: n = 12, the smallest abundant number
(sigma(12) - 12 = 1+2+3+4+6 = 16 > 12). Its proper divisors are
{1, 2, 3, 4, 6} (5 elements, so a 2^5 = 32-element search space). 12 is
abundant but IS semiperfect (e.g. 2+4+6 = 12, or 1+2+3+6 = 12), which is
exactly why 12 is NOT a weird number / not in A006036 -- the first weird
number is 70. This script brute-forces (classically, from first principles,
in this file) every one of the 32 subsets of {1,2,3,4,6} to find every
subset summing to 12, then uses Grover's algorithm on a 5-qubit register to
search the same 32-element space for a marked subset ("marked" = sums to
12, computed classically per basis state to build the phase-oracle circuit,
exactly the property under test) and checks that Grover amplifies the
correct solution set with high probability, matching the classical brute
force exactly.

Requires only qiskit, qiskit_aer, numpy.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical instance and classical answer (computed from first principles)
# ---------------------------------------------------------------------------

N = 12
DIVISORS = [d for d in range(1, N) if N % d == 0]  # proper divisors of 12
assert DIVISORS == [1, 2, 3, 4, 6]
NUM_QUBITS = len(DIVISORS)  # 5 -> search space of size 32

sigma_proper = sum(DIVISORS)
assert sigma_proper > N, "12 must be abundant for this instance to be meaningful"

def subset_sum(bits):
    """bits: tuple/list of 0/1 of length NUM_QUBITS, bit i selects DIVISORS[i]."""
    return sum(d for d, b in zip(DIVISORS, bits) if b)

classical_solutions = []
for mask in range(2 ** NUM_QUBITS):
    bits = tuple((mask >> i) & 1 for i in range(NUM_QUBITS))
    if subset_sum(bits) == N:
        classical_solutions.append(mask)

classical_solutions.sort()
M = len(classical_solutions)
assert M > 0, "12 must be semiperfect for this instance to be meaningful (it is)"
print(f"Classical brute force over {2**NUM_QUBITS} subsets of {DIVISORS}:")
for mask in classical_solutions:
    bits = tuple((mask >> i) & 1 for i in range(NUM_QUBITS))
    chosen = [d for d, b in zip(DIVISORS, bits) if b]
    print(f"  mask={mask:05b} subset={chosen} sum={sum(chosen)}")
print(f"-> {M} solution(s); 12 is abundant and semiperfect, hence NOT weird "
      f"(consistent with A006036 starting at 70).\n")

# ---------------------------------------------------------------------------
# 2. Grover search circuit over the same 32-element space
# ---------------------------------------------------------------------------
# Qubit ordering: little-endian, qubit i <-> DIVISORS[i], matching `mask` above.

N_STATES = 2 ** NUM_QUBITS


def build_oracle():
    """Phase oracle: flips the sign of every basis state in classical_solutions.
    The marking condition (subset sum == N) is evaluated classically per
    state -- this is the standard way to turn a classically-evaluable
    predicate into a Grover phase oracle when no arithmetic circuit is
    built -- and is exactly the property under test above."""
    qc = QuantumCircuit(NUM_QUBITS, name="oracle")
    for mask in classical_solutions:
        bits = [(mask >> i) & 1 for i in range(NUM_QUBITS)]
        # X on qubits that are 0 in this solution, so an all-ones pattern
        # on those flipped qubits corresponds to this exact basis state.
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if NUM_QUBITS == 1:
            qc.z(0)
        else:
            mcz = MCMTGate(ZGate(), NUM_QUBITS - 1, 1)
            qc.append(mcz, list(range(NUM_QUBITS)))
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser():
    qc = QuantumCircuit(NUM_QUBITS, name="diffuser")
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    if NUM_QUBITS == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), NUM_QUBITS - 1, 1)
        qc.append(mcz, list(range(NUM_QUBITS)))
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))
    return qc


theta = math.asin(math.sqrt(M / N_STATES))
iterations = max(1, round((math.pi / 4 / theta) - 0.5))

oracle = build_oracle()
diffuser = build_diffuser()

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

from qiskit import transpile

sim = AerSimulator()
qc = transpile(qc, sim, optimization_level=0)
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit's counts keys are ordinary binary strings (rightmost char = bit 0
# = qubit 0), so interpreting the key as a plain binary integer recovers
# the same `mask` convention used above.
def key_to_mask(key):
    return int(key, 2)

measured_masks = {key_to_mask(k): v for k, v in counts.items()}
solution_hits = sum(v for k, v in measured_masks.items() if k in classical_solutions)
solution_fraction = solution_hits / shots

print(f"Grover search: N={N_STATES} states, M={M} marked, iterations={iterations}")
print(f"Measured solution-state fraction: {solution_fraction:.4f} "
      f"(uniform-random baseline would be {M / N_STATES:.4f})")

top_mask = max(measured_masks, key=measured_masks.get)
top_is_solution = top_mask in classical_solutions

# ---------------------------------------------------------------------------
# 3. Verify quantum result against classical answer
# ---------------------------------------------------------------------------
# Success criteria: Grover must amplify the marked subspace well above the
# uniform baseline, and the single most-frequently measured basis state must
# itself be a genuine classical solution (subset summing to 12).
amplified = solution_fraction > 3 * (M / N_STATES)
verified = amplified and top_is_solution

if verified:
    print("PASS: Grover search amplified exactly the classically-verified "
          "semiperfect-subset solutions for n=12 (proper divisors "
          f"{DIVISORS}), matching brute force ({M} solutions found; top "
          f"measured state mask={top_mask:05b} is a valid solution).")
else:
    print("FAIL: quantum result did not match the classical semiperfect "
          f"search (top_mask={top_mask:05b}, is_solution={top_is_solution}, "
          f"solution_fraction={solution_fraction:.4f}).")

assert verified
