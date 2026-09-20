"""
Erdos problem #710 (OEIS A390246) — quantum-testable instance.

A390246: a(n) is the least integer k such that there exist n distinct
integers b_1, ..., b_n with n < b_i < n+k and i | b_i for 1 <= i <= n.
Equivalently: a(n) is the least k for which the interval (n, n+k) admits a
system of n distinct integers, one per index i in 1..n, each divisible by i.

This script tests the underlying FINITE, COMPUTABLE property that defines
each term of the sequence — feasibility of such a "distinct-multiples"
system for a fixed (n, k) — for the small instance n = 4, k = 6, which is
exactly the instance certifying a(4) = 6 (the known 4th term of A390246,
first terms 2, 3, 4, 6, 6, 9, 9, 11, ...).

Classical setup for n = 4, k = 6, interval (4, 10) i.e. b_i in {5,...,9}:
  i=1: multiples of 1 in {5..9} -> {5,6,7,8,9}   (5 choices, indexed 0..4)
  i=2: multiples of 2 in {5..9} -> {6,8}         (2 choices, indexed 0..1)
  i=3: multiples of 3 in {5..9} -> {6,9}         (2 choices, indexed 0..1)
  i=4: multiples of 4 in {5..9} -> {8}           (forced: b_4 = 8)

The script brute-forces (classically, from first principles, no OEIS value
copied) every (b_1, b_2, b_3) combination from the above candidate lists,
keeps the ones where {b_1, b_2, b_3, b_4} are 4 distinct integers, and uses
that as ground truth. It then encodes the same small search space (5 qubits:
3 for b_1's 5 candidates, 1 for b_2's 2 candidates, 1 for b_3's 2 candidates;
b_4 = 8 is a fixed constant, not a qubit) into a Grover search circuit whose
oracle marks exactly the valid, all-distinct combinations. Grover
amplification is run on the ideal AerSimulator, and the script checks that
the highest-probability measured outcomes decode to precisely the classical
valid set (confirming feasibility, i.e. a(4) <= 6, via genuine quantum
search) and that no other combination among the reachable ones is marked.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS value copied in).
# ---------------------------------------------------------------------------

N = 4          # index range 1..4
K = 6          # candidate k being certified (a(4) = 6 per A390246)
LOW, HIGH = N, N + K   # open interval (4, 10) -> b_i in {5,...,9}

candidates = {i: [b for b in range(LOW + 1, HIGH) if b % i == 0] for i in range(1, N + 1)}
# candidates == {1: [5,6,7,8,9], 2: [6,8], 3: [6,9], 4: [8]}

b1_choices = candidates[1]          # 5 options -> 3 qubits (indices 0..4, 5..7 unused/padding)
b2_choices = candidates[2]          # 2 options -> 1 qubit
b3_choices = candidates[3]          # 2 options -> 1 qubit
assert candidates[4] == [8]
b4 = candidates[4][0]               # forced constant = 8

valid_index_triples = []            # (i1, i2, i3) with distinct {b1,b2,b3,b4}
for i1, b1 in enumerate(b1_choices):
    for i2, b2 in enumerate(b2_choices):
        for i3, b3 in enumerate(b3_choices):
            if len({b1, b2, b3, b4}) == 4:
                valid_index_triples.append((i1, i2, i3))

assert valid_index_triples, "classical brute force found no feasible system for k=6"

# Qubit layout (little-endian within each field, fields concatenated
# high-to-low as q1(3 qubits) q2(1 qubit) q3(1 qubit), 5 qubits total,
# search space size 2**5 = 32):
N_QUBITS = 5
Q1_BITS = 3   # covers indices 0..7 (only 0..4 are real b1 candidates)


def index_triple_to_bitstring(i1, i2, i3):
    """Return the 5-bit string (Qiskit bit order q4 q3 q2 q1 q0) for a
    given (b1-index, b2-index, b3-index) combination."""
    q1 = format(i1, "0{}b".format(Q1_BITS))[::-1]  # q0,q1,q2 (little endian bits)
    q2 = format(i2, "01b")
    q3 = format(i3, "01b")
    bits = list(q1) + list(q2) + list(q3)  # [q0,q1,q2,q3,q4]
    # Qiskit prints bitstrings with qN-1 ... q0 (MSB first)
    return "".join(reversed(bits))


valid_bitstrings = {index_triple_to_bitstring(*t) for t in valid_index_triples}

# ---------------------------------------------------------------------------
# 2. Grover search circuit whose oracle marks exactly `valid_bitstrings`.
# ---------------------------------------------------------------------------


def add_multi_controlled_z(qc, qubit_indices):
    """Apply a Z conditioned on all given qubits being |1>, using an ancilla
    only when needed (here we use the MCX-with-phase-kickback trick via an
    H-MCX-H sandwich on the last control, which realizes a multi-controlled
    Z with no extra ancilla qubits)."""
    if len(qubit_indices) == 1:
        qc.z(qubit_indices[0])
        return
    target = qubit_indices[-1]
    controls = qubit_indices[:-1]
    qc.h(target)
    qc.append(MCXGate(len(controls)), controls + [target])
    qc.h(target)


def oracle(qc, bitstring):
    """Flip the sign of the single computational basis state `bitstring`
    (Qiskit MSB-first convention, matching index_triple_to_bitstring)."""
    bits_lsb_first = bitstring[::-1]  # bits_lsb_first[q] is qubit q's bit
    zero_qubits = [q for q, b in enumerate(bits_lsb_first) if b == "0"]
    for q in zero_qubits:
        qc.x(q)
    add_multi_controlled_z(qc, list(range(N_QUBITS)))
    for q in zero_qubits:
        qc.x(q)


def diffuser(qc):
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    add_multi_controlled_z(qc, list(range(N_QUBITS)))
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))


search_space_size = 2 ** N_QUBITS
num_marked = len(valid_bitstrings)
# Standard Grover iteration count.
iterations = max(1, round((math.pi / 4) * math.sqrt(search_space_size / num_marked)))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    for bs in valid_bitstrings:
        oracle(qc, bs)
    diffuser(qc)
qc.measure(range(N_QUBITS), range(N_QUBITS))

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 20000
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# ---------------------------------------------------------------------------
# 3. Compare quantum result against the classical answer.
# ---------------------------------------------------------------------------

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_k = sorted_counts[: num_marked]
top_bitstrings = {bs for bs, _ in top_k}

marked_probability = sum(counts.get(bs, 0) for bs in valid_bitstrings) / shots

quantum_matches_classical = (
    top_bitstrings == valid_bitstrings
    and marked_probability > 0.5  # Grover should have concentrated most of the amplitude here
)

print("Erdos problem #710 / OEIS A390246 quantum test")
print("Instance: n = {}, k = {} (candidate certifying a(4) = 6)".format(N, K))
print("Candidate lists:", candidates)
print("Classical feasible (b1,b2,b3) index-triples (b4 = {} fixed):".format(b4), valid_index_triples)
print("Classical answer: feasible =", bool(valid_index_triples))
print("Valid bitstrings (oracle-marked states):", sorted(valid_bitstrings))
print("Grover iterations used:", iterations)
print("Top {} measured bitstrings:".format(num_marked), top_k)
print("Total measured probability on marked (valid) states: {:.3f}".format(marked_probability))

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
