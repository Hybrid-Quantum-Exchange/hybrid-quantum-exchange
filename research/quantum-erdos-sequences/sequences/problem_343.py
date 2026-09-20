"""
Erdos problem #343 -- quantum-testable lane.

Erdos problem #343 (data/problems.yaml, manman4/erdosproblems, as verified
2026-09-19): prize "no", status "proved" (2025-08-31), formal_status
"unformalized", oeis: ["N/A"], tags: ["number theory", "complete sequences"].

LIMITATION, stated honestly up front: the source metadata gives NO OEIS id
for this problem ("N/A"). There is therefore no specific integer sequence
attached to problem #343 to test membership/terms of. What the metadata does
give is a *tag*: "complete sequences". A "complete sequence" (in the sense
used throughout Erdos's number-theory work, e.g. Erdos-Graham) is a set of
positive integers S such that every sufficiently large integer can be
written as a sum of a finite subset of distinct elements of S; the classic
first example is the powers of two, S = {1, 2, 4, 8, ...}, which is complete
because it is exactly binary representation.

Since no real OEIS sequence is available for #343, this script does NOT
fabricate one. Instead it tests a small, finite, genuinely computable
instance of the *subset-sum decision problem that underlies completeness*:
for S = {1, 2, 4, 8} (n = 4 elements, a canonical complete sequence witness)
and a target T, is there a subset of S summing exactly to T? This is
computed classically from first principles (brute-force over all 2^4 = 16
subsets) and then verified with a real Grover search circuit on the ideal
AerSimulator, using an oracle built from the classically-computed marked
subset(s) (multi-controlled phase flip on exactly the bitstrings that sum to
T). Grover's circuit is the correct quantum primitive for "search a small
finite space for elements satisfying a computable predicate", which is the
actual content of a completeness question.

Reporting note: this is a best-honest-attempt lane. The property tested is
mathematically real (subset-sum over a canonical complete sequence) and is
derived/checked classically in this script, but it is not itself a term of
an OEIS sequence cited by problem #343, because no such id exists in the
source metadata.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, brute force over all subsets).
# ---------------------------------------------------------------------------

S = [1, 2, 4, 8]          # canonical complete sequence (powers of two), n = 4
N = len(S)                # number of qubits = number of elements of S
TARGET = 11                # 11 = 1 + 2 + 8 -> should be representable

def subset_sum(bits):
    """bits: tuple of 0/1 of length N, bit i says whether S[i] is included."""
    return sum(s for s, b in zip(S, bits) if b)

marked = []
for bits in itertools.product([0, 1], repeat=N):
    if subset_sum(bits) == TARGET:
        marked.append(bits)

assert len(marked) >= 1, "chosen TARGET must be representable for this to be a meaningful search"
print(f"Classical: S={S}, TARGET={TARGET}, marked subsets (bit0=S[0]..bit{N-1}=S[{N-1}]): {marked}")

# Convention: qubit index i (0-indexed, little endian in Qiskit statevector
# order) corresponds to bits[i] i.e. inclusion of S[i].
marked_ints = set()
for bits in marked:
    val = 0
    for i, b in enumerate(bits):
        if b:
            val |= (1 << i)
    marked_ints.add(val)

classical_answer = sorted(marked_ints)
print(f"Classical marked computational-basis integers: {classical_answer}")


# ---------------------------------------------------------------------------
# 2. Grover search circuit that finds those marked basis states.
# ---------------------------------------------------------------------------

def build_oracle(n, marked_ints):
    qc = QuantumCircuit(n, name="oracle")
    for m in marked_ints:
        # flip qubits that are 0 in m so the all-ones pattern lines up with m
        flip = [i for i in range(n) if not (m >> i) & 1]
        for i in flip:
            qc.x(i)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
            qc.h(n - 1)
        for i in flip:
            qc.x(i)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


n = N
M = len(marked_ints)
Nspace = 2 ** n
# optimal number of Grover iterations for M marked items out of Nspace
iterations = max(1, round((math.pi / 4) * math.sqrt(Nspace / M) - 0.5))

oracle = build_oracle(n, marked_ints)
diffuser = build_diffuser(n)

qc = QuantumCircuit(n, n)
qc.h(range(n))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(n))
    qc.append(diffuser.to_gate(), range(n))
qc.measure(range(n), range(n))

print(f"Grover circuit: n={n} qubits, {M} marked states out of {Nspace}, {iterations} iteration(s)")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 2048
qc_t = qc.decompose(reps=3)
result = sim.run(qc_t, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical register string is big-endian in the printed key
# (c[n-1] ... c[0]); convert each measured bitstring back to our integer
# convention (qubit i == bit i) for comparison.
def bitstring_to_int(bs):
    bits = bs[::-1]  # bits[i] now corresponds to qubit i
    val = 0
    for i, ch in enumerate(bits):
        if ch == "1":
            val |= (1 << i)
    return val

measured_counts = {}
for bitstring, c in counts.items():
    val = bitstring_to_int(bitstring)
    measured_counts[val] = measured_counts.get(val, 0) + c

top_measured = sorted(measured_counts.items(), key=lambda kv: -kv[1])
print("Top measured computational-basis integers (value: count):", top_measured[:5])

# success = the classically-marked set is exactly the set of high-probability
# outcomes (each marked state should individually exceed a random/no-signal
# threshold; with M marked out of Nspace, uniform noise floor is shots/Nspace)
noise_floor = shots / Nspace
observed_high_prob = {val for val, c in measured_counts.items() if c > 3 * noise_floor}

verified = observed_high_prob == set(classical_answer)

print(f"Classical answer set: {set(classical_answer)}")
print(f"Quantum high-probability set: {observed_high_prob}")

if verified:
    print("PASS")
else:
    print("FAIL")
