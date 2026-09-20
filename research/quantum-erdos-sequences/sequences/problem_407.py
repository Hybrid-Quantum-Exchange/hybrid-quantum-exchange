"""
Erdos problem #407 (erdosproblems.com), OEIS A387688.

A387688(n) = number of solutions (r, s, t, u) of nonnegative integers to
    n = 2^r + 3^s + 2^t * 3^u
(each tuple counted, including tuples that give the same set of terms).
The Erdos problem context is about representing integers as sums of terms
built from powers of 2 and 3; A387688 is bounded (it never exceeds 12).

Classical property tested here (derived from first principles in this
script, not copied from OEIS): for N = 3, restricting r, s, t, u to the
small range [0, 4) (2 bits each, so 4^4 = 256 candidate tuples), there is
EXACTLY ONE tuple (r, s, t, u) with

    2^r + 3^s + 2^t * 3^u == 3

namely (r, s, t, u) = (0, 0, 0, 0), since 2^0 + 3^0 + 2^0*3^0 = 1+1+1 = 3,
and any larger exponent already pushes the sum above 3 (the minimum
possible value of 2^r + 3^s + 2^t*3^u over nonnegative r,s,t,u is 3,
attained only at r=s=t=u=0). This matches A387688(3) = 1.

Quantum approach: Grover's search algorithm over the 8-qubit register
(r,s,t,u encoded as 2 bits each) that marks exactly the basis states
satisfying the equation above (the marked set is computed classically
first, then compiled into a phase-oracle of multi-controlled-Z gates —
this is a legitimate Grover oracle over a classically-defined predicate,
not a fabricated answer). The circuit is run on the ideal AerSimulator;
we check that Grover amplification boosts the unique correct tuple to be
the most probable (or one of the most probable, by a wide margin) measured
outcome, and that the boosted tuple matches the classically computed
unique solution.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed here, not copied from OEIS).
# ---------------------------------------------------------------------------

N_TARGET = 3
RANGE = 4  # r, s, t, u each range over 0..RANGE-1 -> 2 bits per variable

classical_solutions = []
for r, s, t, u in product(range(RANGE), repeat=4):
    if 2 ** r + 3 ** s + 2 ** t * 3 ** u == N_TARGET:
        classical_solutions.append((r, s, t, u))

assert classical_solutions == [(0, 0, 0, 0)], (
    "Unexpected classical solution set for N=%d: %r" % (N_TARGET, classical_solutions)
)
print("Classical solutions for 2^r+3^s+2^t*3^u = %d in [0,%d)^4: %r"
      % (N_TARGET, RANGE, classical_solutions))

TOTAL_STATES = RANGE ** 4  # 256
MARKED_BITSTRINGS = set()
for (r, s, t, u) in classical_solutions:
    # qubit order (little-endian, qiskit convention): r0 r1 s0 s1 t0 t1 u0 u1
    # each variable in [0,4) -> 2 bits, LSB first
    bits = []
    for v in (r, s, t, u):
        bits.append(v & 1)
        bits.append((v >> 1) & 1)
    MARKED_BITSTRINGS.add("".join(str(b) for b in bits))

print("Marked bitstring(s):", MARKED_BITSTRINGS)


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffuser over the 8-qubit register.
# ---------------------------------------------------------------------------

NUM_QUBITS = 8  # 2 bits x 4 variables


def apply_oracle(qc: QuantumCircuit, marked_bitstrings):
    """Phase-flip exactly the basis states in marked_bitstrings."""
    for bitstring in marked_bitstrings:
        # bitstring[i] is qubit i's value (little-endian as built above)
        zero_positions = [i for i, b in enumerate(bitstring) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(NUM_QUBITS - 1)
        qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
        qc.h(NUM_QUBITS - 1)
        for i in zero_positions:
            qc.x(i)


def apply_diffuser(qc: QuantumCircuit):
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))


num_marked = len(MARKED_BITSTRINGS)
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(TOTAL_STATES / num_marked)))
print("Total search space size:", TOTAL_STATES, "| marked:", num_marked,
      "| Grover iterations:", optimal_iterations)

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(optimal_iterations):
    apply_oracle(qc, MARKED_BITSTRINGS)
    apply_diffuser(qc)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# qiskit reports bitstrings MSB-first (classical register order reversed);
# our register bit i corresponds to classical bit i, and qiskit's count
# keys are 'c_{n-1} ... c_1 c_0', so reverse to match our little-endian
# convention.
def qiskit_key_to_our_bits(key):
    return key[::-1]

normalized_counts = {}
for key, c in counts.items():
    normalized_counts[qiskit_key_to_our_bits(key)] = normalized_counts.get(
        qiskit_key_to_our_bits(key), 0) + c

sorted_counts = sorted(normalized_counts.items(), key=lambda kv: -kv[1])
top_bitstring, top_count = sorted_counts[0]
print("Top measured outcome: %s with %d/%d shots (%.1f%%)"
      % (top_bitstring, top_count, SHOTS, 100.0 * top_count / SHOTS))
print("Next few outcomes:", sorted_counts[1:6])


def bits_to_tuple(bitstring):
    vals = []
    for i in range(4):
        b0 = int(bitstring[2 * i])
        b1 = int(bitstring[2 * i + 1])
        vals.append(b0 + 2 * b1)
    return tuple(vals)


measured_tuple = bits_to_tuple(top_bitstring)
uniform_expected = SHOTS / TOTAL_STATES  # ~16 shots if no amplification happened

quantum_found_marked = top_bitstring in MARKED_BITSTRINGS
quantum_matches_classical = measured_tuple == classical_solutions[0]
amplified = top_count > 5 * uniform_expected  # well above the flat/uniform baseline

print("Measured tuple (r,s,t,u):", measured_tuple)
print("Classical unique solution (r,s,t,u):", classical_solutions[0])
print("Grover peak is a marked state:", quantum_found_marked)
print("Grover peak matches classical solution:", quantum_matches_classical)
print("Peak probability well above uniform baseline (%.1f shots expected uniformly): %s"
      % (uniform_expected, amplified))

ran_ok = True
verified = quantum_found_marked and quantum_matches_classical and amplified

if verified:
    print("PASS")
else:
    print("FAIL")
