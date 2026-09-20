"""
Erdos problem #862 (erdosproblems.com), oeis id A382395.

A382395: "Number of maximum sized subsets of {1..n} such that every pair of
distinct elements has a different difference" -- i.e. the number of maximum
Sidon sets (also called B2 sets / Sidon sets: sets where all pairwise
differences a_i - a_j, i != j, are distinct) contained in {1, ..., n}.
A382395(4) = 2.

Classical property tested here (computed from first principles below, not
copied from OEIS): for n = 5, k = 3 (the maximum Sidon-set size in {1..5}),
enumerate all C(5,3) = 10 subsets of {1,...,5} of size 3, and mark exactly
those subsets that are Sidon sets (all pairwise differences distinct). This
script:
  1. Computes classically which of the 10 subsets are Sidon sets, and checks
     that the count matches A382395(5) = 6 (OEIS b-file: offset 0, a(5)=6)
     -- this is the ground truth used below.
  2. Builds a genuine 4-qubit (16 basis states) Grover search circuit whose
     oracle marks exactly those subset-indices (derived from step 1,
     hardcoded into the oracle as phase flips on the matching computational
     basis states -- indices 10..15 are unused padding and never marked),
     and amplifies them with one Grover diffusion step (the optimal
     iteration count for N=16, M=6).
  3. Runs the circuit on the ideal AerSimulator, and checks that measured
     outcomes concentrate, well above the uniform baseline of M/N = 6/16 =
     0.375, on subset-indices that are, when independently re-checked
     classically, exactly the Sidon subsets found in step 1.
  4. Prints PASS if the quantum search's measured distribution is
     concentrated on (only) the classical Sidon-set indices with
     probability well above baseline, and their count equals A382395(5);
     otherwise prints FAIL.

(Note: n=4 -- the OEIS page's own example, A382395(4)=2 -- was tried first
but rejected: with N=4 basis states and M=2 marked, Grover search is exactly
the M=N/2 degenerate case where a single iteration provably does not change
the 50% baseline success probability, so it cannot demonstrate genuine
amplitude amplification. n=5 avoids that degeneracy.)
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# Step 1: classical computation, from first principles.
# ---------------------------------------------------------------------------

N_VALUE = 5          # the n in A382395(n)
SUBSET_SIZE = 3       # the maximum Sidon-set size in {1..5}
EXPECTED_A382395_5 = 6  # OEIS A382395(5) (b-file, offset 0), verified below

universe = list(range(1, N_VALUE + 1))  # {1, 2, 3, 4, 5}
subsets = list(itertools.combinations(universe, SUBSET_SIZE))  # C(5,3) = 10 subsets


def is_sidon(subset):
    """A subset is a Sidon set iff all pairwise (positive) differences
    among its elements are distinct."""
    diffs = []
    for a, b in itertools.combinations(subset, 2):
        diffs.append(abs(a - b))
    return len(diffs) == len(set(diffs))


sidon_flags = [is_sidon(s) for s in subsets]
sidon_indices = sorted(i for i, flag in enumerate(sidon_flags) if flag)
sidon_subsets = [subsets[i] for i in sidon_indices]

print("Subsets of {1,2,3,4,5} of size 3, in enumeration order:")
for i, s in enumerate(subsets):
    print(f"  index {i} ({i:02b}): {s}  Sidon={sidon_flags[i]}")

print(f"\nClassically found Sidon subsets: {sidon_subsets}")
print(f"Count = {len(sidon_subsets)}, expected A382395({N_VALUE}) = {EXPECTED_A382395_5}")

assert len(subsets) == 10, "expected exactly 10 subsets (C(5,3))"
assert len(sidon_subsets) == EXPECTED_A382395_5, (
    "classical Sidon count does not match OEIS A382395(5); refusing to build "
    "a circuit around a wrong ground truth"
)

# ---------------------------------------------------------------------------
# Step 2: build a real 4-qubit Grover search circuit whose oracle marks
# exactly `sidon_indices` (computed purely classically above). Indices
# 10..15 are unused padding (never marked) since 10 subsets do not fill all
# 16 basis states of 4 qubits.
# ---------------------------------------------------------------------------

NUM_QUBITS = 4  # log2(16) states; subset indices 0..9 are meaningful
N = 2 ** NUM_QUBITS
M = len(sidon_indices)


def mark_state(qc, index, num_qubits):
    """Apply an X on each qubit whose bit is 0 in `index`, so that the
    all-ones pattern (for a multi-controlled Z below) corresponds exactly
    to computational basis state |index>."""
    bits = format(index, f"0{num_qubits}b")
    for q, bit in enumerate(reversed(bits)):
        if bit == "0":
            qc.x(q)


def oracle(qc, indices, num_qubits):
    for idx in indices:
        mark_state(qc, idx, num_qubits)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        mark_state(qc, idx, num_qubits)  # uncompute the X's


def diffuser(qc, num_qubits):
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))


# Optimal number of Grover iterations for N states, M marked.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    oracle(qc, sidon_indices, NUM_QUBITS)
    diffuser(qc, NUM_QUBITS)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

# ---------------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

SHOTS = 4096
backend = AerSimulator()
compiled = transpile(qc, backend)
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

print(f"\nGrover iterations used: {iterations}")
print(f"Measurement counts (bitstring -> count): {counts}")

# Qiskit's classical-bit string is written with qubit (n-1) leftmost; our
# indices were encoded with qubit 0 as the least-significant bit, so the
# measured bitstring, read as an unsigned binary integer, equals the index.
measured_indices = {int(bitstring, 2): count for bitstring, count in counts.items()}

# ---------------------------------------------------------------------------
# Step 4: verify quantum result against the classical answer.
# ---------------------------------------------------------------------------

total_marked_shots = sum(c for idx, c in measured_indices.items() if idx in sidon_indices)
success_probability = total_marked_shots / SHOTS
baseline_probability = M / N  # what an unamplified uniform search would give

# The outcome(s) receiving the most shots should be exactly the marked
# (Sidon) indices, each independently re-verified classically, and none of
# the top-M outcomes should be an unmarked/padding index.
sorted_by_count = sorted(measured_indices.items(), key=lambda kv: -kv[1])
top_outcomes = {idx for idx, c in sorted_by_count[:M]}

reverified_sidon = {i for i in top_outcomes if i < len(subsets) and is_sidon(subsets[i])}

theoretical_success_probability = math.sin((2 * iterations + 1) * theta) ** 2

quantum_matches_classical = (
    top_outcomes == set(sidon_indices)
    and reverified_sidon == top_outcomes
    and success_probability > baseline_probability + 0.25  # genuine amplification
    and success_probability > 0.75
)

print(f"\nTop {M} measured outcome index(es): {sorted(top_outcomes)}")
print(f"Classical Sidon index(es):          {sorted(sidon_indices)}")
print(f"Baseline (unamplified) success probability: {baseline_probability:.4f}")
print(f"Theoretical Grover success probability:      {theoretical_success_probability:.4f}")
print(f"Measured success probability:                 {success_probability:.4f}")

if quantum_matches_classical:
    print("\nPASS")
else:
    print("\nFAIL")
