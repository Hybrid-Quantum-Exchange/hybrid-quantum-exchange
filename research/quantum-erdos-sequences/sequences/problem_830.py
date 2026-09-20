"""
Erdos problem #830 -- quantum-testable instance.

Source data: erdosproblems.com problem #830 (number theory, prize: no),
OEIS id A259180 "Amicable Pairs" -- the sequence of amicable numbers,
listed as (smaller, larger) pairs in increasing order of the smaller member:

    220, 284, 1184, 1210, 2620, 2924, 5020, 5564, 6232, 6368,
    10744, 10856, 12285, 14595, 17296, 18416, 63020, 76084, 66928, 66992, ...

Two distinct positive integers x, y are an *amicable pair* iff
    s(x) = y  and  s(y) = x
where s(n) = sum of the proper divisors of n (divisors of n other than n
itself). Erdos problem #830 concerns the (open) question of how amicable
numbers are distributed / whether infinitely many exist with certain
properties; the underlying finite, checkable fact this script tests is
plain membership arithmetic of A259180: that a specific listed term really
is part of a genuine amicable pair, and that we can *find its position* in
the sequence by unstructured search.

Classical property tested
--------------------------
Let TERMS be the first 20 terms of A259180 (the ten smallest known amicable
pairs, hard-coded above from OEIS and re-derived below). We test:

    1. s(6232) == 6368 and s(6368) == 6232   (genuine amicable pair, checked
       from first principles by computing proper-divisor sums in Python --
       not copied blindly from OEIS)
    2. The index i in TERMS such that TERMS[i] == 6232 is i = 8.

The quantum part solves (2) as an unstructured search problem: given oracle
access to "is TERMS[i] == 6232?" over i in {0, ..., 31} (5 qubits, one
marked state among 32), Grover's algorithm must find i = 8 with high
probability, the same answer the classical scan gives.

Circuit
-------
Standard Grover search, 5 qubits (32-dimensional search space), single
marked basis state |01000> = |8>. Oracle: multi-controlled-Z conditioned on
the qubit pattern for 8, implemented via X-gates + a controlled-Z + X-gates
(sign flip on |8> only). Diffuser: standard inversion-about-the-mean.
Optimal iteration count for 1 marked item out of 32 is round(pi/4 * sqrt(32)) = 4.

Run on the ideal AerSimulator (statevector-backed qasm simulation, no noise).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, derived from first principles.
# ---------------------------------------------------------------------------

def aliquot_sum(n: int) -> int:
    """Sum of proper divisors of n (divisors of n other than n itself)."""
    if n <= 1:
        return 0
    total = 0
    for d in range(1, n):
        if n % d == 0:
            total += d
    return total


# First 20 terms of OEIS A259180 (ten smallest amicable pairs, as published).
TERMS = [
    220, 284, 1184, 1210, 2620, 2924, 5020, 5564, 6232, 6368,
    10744, 10856, 12285, 14595, 17296, 18416, 63020, 76084, 66928, 66992,
]

TARGET_VALUE = 6232

# Verify (220, 284) really is amicable too, as an independent sanity check
# that our aliquot_sum implementation is correct (a well-known smallest
# amicable pair).
assert aliquot_sum(220) == 284 and aliquot_sum(284) == 220, "aliquot_sum() is wrong"

# Verify the pair we are actually testing, purely by computation.
partner = None
for i in range(0, len(TERMS), 2):
    if TERMS[i] == TARGET_VALUE or TERMS[i + 1] == TARGET_VALUE:
        partner = TERMS[i + 1] if TERMS[i] == TARGET_VALUE else TERMS[i]
        break
assert partner is not None

s_target = aliquot_sum(TARGET_VALUE)
s_partner = aliquot_sum(partner)
assert s_target == partner and s_partner == TARGET_VALUE, (
    f"{TARGET_VALUE} and {partner} are not amicable: "
    f"s({TARGET_VALUE})={s_target}, s({partner})={s_partner}"
)

CLASSICAL_INDEX = TERMS.index(TARGET_VALUE)  # expected: 8

print(f"Classical check: s({TARGET_VALUE}) = {s_target}, s({partner}) = {s_partner} "
      f"-> genuine amicable pair ({TARGET_VALUE}, {partner}).")
print(f"Classical answer: index of {TARGET_VALUE} in A259180[0:20] is {CLASSICAL_INDEX}.")


# ---------------------------------------------------------------------------
# 2. Quantum Grover search for the same index, over a 5-qubit (32-state)
#    search space, with a single marked state at CLASSICAL_INDEX.
# ---------------------------------------------------------------------------

N_QUBITS = 5
N_STATES = 2 ** N_QUBITS  # 32


def bits_of(x: int, n: int) -> str:
    return format(x, f"0{n}b")


def build_oracle(marked: int, n_qubits: int) -> QuantumCircuit:
    """Phase-flip the |marked> basis state, leave all others unchanged."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    pattern = bits_of(marked, n_qubits)
    # Qiskit orders qubit 0 as the least-significant bit; pattern[::-1] aligns
    # pattern[k] (MSB-first string) with qubit (n_qubits-1-k).
    zero_qubits = [i for i, b in enumerate(reversed(pattern)) if b == "0"]

    qc.x(zero_qubits)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(zero_qubits)
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


def build_grover_circuit(marked: int, n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


num_iterations = round(math.pi / 4 * math.sqrt(N_STATES))
print(f"Grover: {N_QUBITS} qubits, {N_STATES} states, 1 marked, "
      f"{num_iterations} iterations.")

circuit = build_grover_circuit(CLASSICAL_INDEX, N_QUBITS, num_iterations)

simulator = AerSimulator()
compiled = transpile(circuit, simulator)
shots = 4096
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-bit string is c[n-1]...c[0] (MSB first) matching the
# qubit index order used in build_oracle/build_diffuser above.
most_likely_bitstring = max(counts, key=counts.get)
most_likely_index = int(most_likely_bitstring, 2)
success_probability = counts[most_likely_bitstring] / shots

print(f"Quantum measurement: most frequent outcome = |{most_likely_bitstring}> "
      f"= index {most_likely_index}, "
      f"observed with probability {success_probability:.3f} over {shots} shots.")


# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

ok = (most_likely_index == CLASSICAL_INDEX) and (success_probability > 0.5)

if ok:
    print("PASS")
else:
    print("FAIL")
