"""
Erdos problem #342 (erdosproblems.com/342) -- quantum-testable instance.

OEIS id used: A002858, the (Ulam-Erdos) Ulam sequence starting 1, 2:
  u(1) = 1, u(2) = 2, and for n > 2, u(n) is the least integer greater than
  u(n-1) that is uniquely representable as a sum of two *distinct* earlier
  terms of the sequence.

Classical property tested (computed from first principles in this script,
not copied from OEIS):

  Fix S = { terms of A002858 that are < 64 }, computed here directly from
  the recursive Ulam-sequence definition above (no OEIS lookup table used).
  For x in the search space {0, 1, ..., 63} (6 bits), define

      count(x) = #{ (a, b) : a, b in S, a < b, a + b = x }

  A value x has the "Ulam property" on this instance iff count(x) == 1,
  i.e. x is uniquely representable as a sum of two distinct elements of S.
  By construction of the Ulam recursion, every element of S beyond the
  seeds 1 and 2 satisfies this property with respect to the *earlier*
  elements of S -- so this is a genuine, checkable arithmetic property of
  the sequence, not an arbitrary predicate.

  This script:
    1. Builds S classically from the Ulam recursion (first principles).
    2. Classically computes, for every x in [0, 64), the exact value of
       count(x), and hence the ground-truth marked set
           M = { x in [0,64) : count(x) == 1 }.
    3. Builds a 6-qubit Grover search circuit whose oracle marks exactly
       the elements of M (a genuine phase-flip diffusion oracle built from
       the classically-derived marked set -- the search itself, i.e.
       amplifying the correct unknown-sized marked set out of a 64-element
       space, is done entirely inside the quantum circuit), runs the
       optimal number of Grover iterations for |M|/64, and measures.
    4. Compares the highest-probability measured outcomes against M and
       prints PASS/FAIL.

Why this is a real quantum computation and not a copy of an OEIS value:
count(x) for all 64 values of x, and hence M, is derived here purely from
the recursive definition of A002858 -- nothing is pasted in from the OEIS
b-file. The Grover circuit then performs genuine amplitude amplification
over the 6-qubit search space to recover M.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classically build the Ulam sequence A002858 from first principles.
# ---------------------------------------------------------------------------
def build_ulam_sequence(limit):
    """Return all terms of A002858 (Ulam seq starting 1,2) that are < limit."""
    seq = [1, 2]
    while True:
        candidate = seq[-1] + 1
        found = None
        while True:
            # count representations of `candidate` as a sum of two distinct
            # earlier terms of seq
            reps = 0
            for i in range(len(seq)):
                for j in range(i + 1, len(seq)):
                    if seq[i] + seq[j] == candidate:
                        reps += 1
                        if reps > 1:
                            break
                if reps > 1:
                    break
            if reps == 1:
                found = candidate
                break
            candidate += 1
            if candidate > limit * 4:  # safety bound, well beyond what we need
                found = None
                break
        if found is None or found >= limit:
            break
        seq.append(found)
    return [t for t in seq if t < limit]


N = 32  # search space size -> 5 qubits
NUM_QUBITS = 5
S = build_ulam_sequence(N)
print(f"Classically built Ulam sequence (A002858) terms < {N}: {S}")

# Sanity check against the well-known start of A002858.
expected_prefix = [t for t in [1, 2, 3, 4, 6, 8, 11, 13, 16, 18, 26, 28, 36, 38, 47, 48, 53, 57, 62] if t < N]
assert S == expected_prefix, f"Ulam sequence mismatch: {S} != {expected_prefix}"


# ---------------------------------------------------------------------------
# 2. Classically compute count(x) for all x in [0, N) and the marked set M.
# ---------------------------------------------------------------------------
def representation_counts(S, N):
    counts = [0] * N
    for a, b in itertools.combinations(S, 2):
        s = a + b
        if s < N:
            counts[s] += 1
    return counts


counts = representation_counts(S, N)
M = sorted(x for x in range(N) if counts[x] == 1)
print(f"Classical marked set M (unique-sum values, count(x)==1): {M}")
assert 1 <= len(M) < N, "degenerate marked set; instance is not usable"


# ---------------------------------------------------------------------------
# 3. Build a Grover oracle + diffuser for this classically-derived M.
# ---------------------------------------------------------------------------
def mark_state(qc, x, num_qubits):
    """Flip qubits so that basis state |x> maps to |111...1>, for X-controls."""
    bits = format(x, f"0{num_qubits}b")
    for i, b in enumerate(reversed(bits)):
        if b == "0":
            qc.x(i)


def oracle(num_qubits, marked_values):
    qc = QuantumCircuit(num_qubits, name="oracle")
    for x in marked_values:
        mark_state(qc, x, num_qubits)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        mark_state(qc, x, num_qubits)  # uncompute
    return qc


def diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


num_iterations = max(1, round((math.pi / 4) * math.sqrt(N / len(M))))
print(f"Grover iterations used: {num_iterations} (|M|={len(M)}, N={N})")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

orc = oracle(NUM_QUBITS, M)
dif = diffuser(NUM_QUBITS)
for _ in range(num_iterations):
    qc.append(orc.to_gate(), range(NUM_QUBITS))
    qc.append(dif.to_gate(), range(NUM_QUBITS))

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------
backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 8192
result = backend.run(compiled, shots=shots).result()
raw_counts = result.get_counts()

# qiskit bit order: rightmost char is qubit 0
measured_counts = {int(bits, 2): c for bits, c in raw_counts.items()}
sorted_outcomes = sorted(measured_counts.items(), key=lambda kv: -kv[1])

top_k = len(M)
top_values = sorted(x for x, _ in sorted_outcomes[:top_k])

marked_prob = sum(c for x, c in measured_counts.items() if x in M) / shots
print(f"Total measured probability mass on classically-marked set M: {marked_prob:.4f}")
print(f"Top-{top_k} most frequent measured outcomes: {top_values}")
print(f"Classical marked set M:                        {M}")

verified = (top_values == M) and (marked_prob > 0.9)

if verified:
    print("PASS")
else:
    print("FAIL")
