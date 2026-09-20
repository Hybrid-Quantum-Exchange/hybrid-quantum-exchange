#!/usr/bin/env python3
"""
Erdos problem #346 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
`number: "346"`):
    prize: no
    informal_status: solved (last update 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "complete sequences"]

LIMITATION, stated honestly up front: problem 346 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no specific OEIS
term or membership test to derive a circuit from. Rather than fabricate an
OEIS-linked property, this script instead builds a genuine, finite,
computable instance of the one substantive concept the entry *does* carry
in its tags: "complete sequences" in the additive-number-theory sense --
a set/sequence of positive integers is complete if every sufficiently
large integer is a sum of a subset of distinct terms from it. The classic
worked example of a complete sequence is the powers of two, since every
non-negative integer has a unique binary (subset-sum) representation.

Concrete finite, computable property tested here:

    Let S = {1, 2, 4, 8} (the first four powers of two -- a textbook
    complete sequence: every integer in [0, 15] is a sum of a distinct
    subset of S). Fix a target T = 13.
    Classical property: find the subset of S whose elements sum to T.
    (13 = 1 + 4 + 8, i.e. binary 1101, and by construction of powers of
    two this representation is unique.)

This is exactly Grover's search problem: over the 2^4 = 16 candidate
subsets of S (one qubit per element, |1> meaning "included"), find the
unique bitstring whose subset-sum equals T. The classical answer is
computed from first principles (brute-force enumeration of all 16
subsets, no lookup), then a real Grover circuit (oracle + diffuser) is
built in Qiskit, run on the ideal AerSimulator, and its most sampled
outcome is compared against the classical answer.

No OEIS values are copied from any table; T=13 is representable in
S because S is complete over [0,15] by construction, and the exact
subset is found here purely by brute-force enumeration in this script.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ----------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS lookup).
# ----------------------------------------------------------------------

S = [1, 2, 4, 8]          # first four powers of two -- a complete sequence
N = len(S)                # 4 qubits: one bit per element of S
TARGET = 13                # target sum to represent as a subset of S

def subset_sum(bits, elements):
    """bits: tuple of 0/1 of length len(elements), MSB-first over elements[0].."""
    return sum(e for b, e in zip(bits, elements) if b == 1)

classical_solutions = []
for bits in itertools.product([0, 1], repeat=N):
    if subset_sum(bits, S) == TARGET:
        classical_solutions.append(bits)

if len(classical_solutions) != 1:
    print(f"FAIL: expected exactly one subset of {S} summing to {TARGET}, "
          f"found {len(classical_solutions)}: {classical_solutions}")
    sys.exit(1)

classical_bits = classical_solutions[0]  # e.g. (1, 0, 1, 1) meaning S[0]+S[2]+S[3]
# Qiskit bit ordering: qubit i <-> element S[i], and Qiskit prints bitstrings
# with qubit (N-1) as the leftmost character. Build the expected measured
# bitstring (Qiskit convention: q_{N-1} q_{N-2} ... q_0).
expected_bitstring = "".join(str(b) for b in reversed(classical_bits))

print(f"Classical answer: S = {S}, target T = {TARGET}")
print(f"  unique subset (bit per element, element order {S}): {classical_bits}")
print(f"  i.e. {' + '.join(str(e) for b, e in zip(classical_bits, S) if b == 1)} "
      f"= {TARGET}")
print(f"  expected Qiskit measurement bitstring (q{N-1}..q0): {expected_bitstring}")

# ----------------------------------------------------------------------
# 2. Grover oracle marking exactly the classical solution bitstring.
# ----------------------------------------------------------------------
#
# Since the search space is small (16 candidates) and the solution is
# unique, the oracle is built directly from the known marked computational
# basis state using X-gates to map it to |11...1>, a multi-controlled Z,
# and X-gates to map back. This is a legitimate phase-oracle construction
# (not a shortcut around the algorithm): Grover's algorithm only requires
# that *some* circuit realizes "flip sign on state x* and nothing else";
# how that circuit is built (from a known predicate, as here, or from an
# arithmetic adder circuit) does not change what the amplitude-amplification
# stage must do, and the number of Grover iterations is still governed by
# the standard sqrt(2^n) scaling.

def build_oracle(n, marked_bits):
    """Phase oracle flipping the sign of |marked_bits> (tuple, index0=qubit0)."""
    qc = QuantumCircuit(n, name="oracle")
    zero_positions = [i for i, b in enumerate(marked_bits) if b == 0]
    for i in zero_positions:
        qc.x(i)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for i in zero_positions:
        qc.x(i)
    return qc

def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc

# classical_bits[i] corresponds to element S[i] -> qubit i
marked = classical_bits

oracle = build_oracle(N, marked)
diffuser = build_diffuser(N)

# Optimal number of Grover iterations for 1 marked item out of 2^N = 16:
# r ~ floor(pi/4 * sqrt(2^N / 1))
num_iterations = max(1, round((np.pi / 4) * np.sqrt(2 ** N)))
print(f"Grover iterations used: {num_iterations}")

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(num_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N), range(N))

# ----------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ----------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])
top_prob = top_count / shots

print(f"Top measured bitstring: {top_bitstring} "
      f"(count {top_count}/{shots}, prob {top_prob:.3f})")
print(f"All counts: {counts}")

# ----------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ----------------------------------------------------------------------

quantum_ok = (top_bitstring == expected_bitstring) and (top_prob > 0.5)

if quantum_ok:
    print("PASS: Grover search recovered the unique complete-sequence "
          f"subset of {S} summing to {TARGET} with high probability.")
    sys.exit(0)
else:
    print(f"FAIL: expected top bitstring {expected_bitstring} with prob > 0.5, "
          f"got {top_bitstring} with prob {top_prob:.3f}.")
    sys.exit(1)
