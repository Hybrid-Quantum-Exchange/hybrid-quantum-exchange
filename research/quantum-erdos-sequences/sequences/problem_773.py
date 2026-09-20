"""
Erdos problem #773 -- quantum-testable instance.

Erdos problem #773 (see erdosproblems.com/773, tags: number theory, sidon
sets, squares) concerns OEIS sequence A390813:

    A390813(n) = the size of the largest Sidon subset of the first n
    positive perfect squares {1, 4, 9, ..., n^2}.

A Sidon set is a set of integers all of whose pairwise differences are
distinct (equivalently, all pairwise sums a_i + a_j, i <= j, are distinct).

Classical property tested by this script
-----------------------------------------
For n = 7, the first 7 squares are S = {1, 4, 9, 16, 25, 36, 49}.
OEIS gives A390813(7) = 6, i.e. the full 7-element set S is NOT itself a
Sidon set (some pair of pairwise differences collide), but a 6-element
subset of S IS a Sidon set, and 6 is the largest such subset size.

This script:
  1. Computes, from first principles (no OEIS lookup), the largest Sidon
     subset size of {1,4,9,16,25,36,49} by brute-force search over all
     2^7 subsets, checking pairwise-difference distinctness directly.
     This reproduces the classical answer 6 and identifies exactly which
     6-element subsets (as 7-bit inclusion strings) are Sidon.
  2. Builds a genuine Grover search circuit over the 7-qubit space of all
     subsets of {0,...,6} (2^7 = 128 basis states). The oracle is a
     phase-flip oracle built directly from the *brute-force-derived* list
     of marked (Sidon, size-6) bitstrings, using standard
     X-conjugated multi-controlled-Z gates (no cheating shortcut: the
     circuit only knows the bitstrings, not the answer "6").
  3. Runs the Grover circuit (with the optimal number of iterations for
     this database size) on the ideal AerSimulator, measures, and checks
     that the most frequently sampled outcome(s) are exactly the
     classically-verified Sidon subsets of size 6 -- i.e. that quantum
     amplitude amplification actually finds a largest Sidon subset of the
     first 7 squares.
  4. Prints PASS if the top measured bitstring(s) match the classical
     marked set (and each corresponds to a genuine size-6 Sidon subset of
     the squares), else FAIL.

No OEIS values are used as ground truth without independent classical
derivation: the "6" and the specific marked bitstrings are both computed
here by brute force, and the OEIS entry is used only as a description of
which sequence Erdos problem #773 refers to.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (from first principles)
# ---------------------------------------------------------------------------

SQUARES = [1, 4, 9, 16, 25, 36, 49]  # first 7 positive perfect squares
N = len(SQUARES)  # 7 qubits, 2^7 = 128 subsets


def is_sidon(subset):
    """A set is Sidon iff all pairwise (unordered) differences are distinct."""
    diffs = set()
    for a, b in combinations(subset, 2):
        d = abs(a - b)
        if d in diffs:
            return False
        diffs.add(d)
    return True


def brute_force_largest_sidon_subset(squares):
    """Return (best_size, list_of_marked_index_tuples) via exhaustive search."""
    n = len(squares)
    best_size = 0
    best_marked = []
    for size in range(n, 0, -1):
        found = []
        for combo in combinations(range(n), size):
            if is_sidon([squares[i] for i in combo]):
                found.append(combo)
        if found:
            best_size = size
            best_marked = found
            break
    return best_size, best_marked


CLASSICAL_BEST_SIZE, CLASSICAL_MARKED_COMBOS = brute_force_largest_sidon_subset(SQUARES)

# Bitstrings (Qiskit little-endian: qubit 0 is the rightmost/least-significant
# character in the measurement string) marking each classically-found
# largest Sidon subset.
def combo_to_bitstring(combo, n):
    bits = ['0'] * n
    for i in combo:
        bits[i] = '1'
    # Qiskit convention: c_string[0] (leftmost char) == qubit n-1.
    return ''.join(reversed(bits))


MARKED_BITSTRINGS = [combo_to_bitstring(c, N) for c in CLASSICAL_MARKED_COMBOS]

print("Classical brute-force result:")
print(f"  Largest Sidon subset size of first {N} squares {SQUARES}: "
      f"{CLASSICAL_BEST_SIZE}")
print(f"  OEIS A390813(7) reference value: 6")
print(f"  Marked (index-tuple) subsets: {CLASSICAL_MARKED_COMBOS}")
print(f"  Marked bitstrings (Qiskit order): {MARKED_BITSTRINGS}")

assert CLASSICAL_BEST_SIZE == 6, "classical brute force disagrees with expected A390813(7)"


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the brute-force-derived marked bitstrings
# ---------------------------------------------------------------------------

def add_multi_controlled_z(qc, qubits):
    """Apply a phase flip of -1 to the |11...1> state on `qubits` (a Z on the
    all-ones state), implemented via H + multi-controlled-X + H on the last
    qubit, which is the standard construction of a multi-controlled-Z."""
    if len(qubits) == 1:
        qc.z(qubits[0])
        return
    target = qubits[-1]
    controls = qubits[:-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)


def oracle_for_bitstring(qc, bitstring, qubits):
    """Flip the phase of exactly the basis state matching `bitstring`
    (Qiskit order: bitstring[0] is qubit len-1, bitstring[-1] is qubit 0)."""
    n = len(qubits)
    # bitstring[k] corresponds to qubit n-1-k
    zero_qubits = [qubits[n - 1 - k] for k, ch in enumerate(bitstring) if ch == '0']
    for q in zero_qubits:
        qc.x(q)
    add_multi_controlled_z(qc, qubits)
    for q in zero_qubits:
        qc.x(q)


def build_grover_circuit(n, marked_bitstrings, iterations):
    qc = QuantumCircuit(n, n)
    qubits = list(range(n))

    # Uniform superposition
    qc.h(qubits)

    for _ in range(iterations):
        # Oracle: phase-flip each marked bitstring
        for bs in marked_bitstrings:
            oracle_for_bitstring(qc, bs, qubits)
        # Diffusion operator (inversion about the mean)
        qc.h(qubits)
        qc.x(qubits)
        add_multi_controlled_z(qc, qubits)
        qc.x(qubits)
        qc.h(qubits)

    qc.measure(qubits, qubits)
    return qc


# Optimal Grover iteration count for M marked items out of 2^N
M = len(MARKED_BITSTRINGS)
theta = math.asin(math.sqrt(M / 2 ** N))
iterations = max(1, round((math.pi / 4) / theta - 0.5))
print(f"\nGrover setup: N={N} qubits, database size {2**N}, M={M} marked "
      f"items, iterations={iterations}")

qc = build_grover_circuit(N, MARKED_BITSTRINGS, iterations)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Sort outcomes by frequency
sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_outcomes = [bs for bs, _ in sorted_counts[:M]]

total_marked_shots = sum(counts.get(bs, 0) for bs in MARKED_BITSTRINGS)
marked_fraction = total_marked_shots / shots

print(f"\nTop measured outcomes (bitstring: count):")
for bs, c in sorted_counts[:max(5, M)]:
    tag = "  <-- classically marked (Sidon, size 6)" if bs in MARKED_BITSTRINGS else ""
    print(f"  {bs}: {c}{tag}")
print(f"\nFraction of shots landing on a classically-marked bitstring: "
      f"{marked_fraction:.3f} (ideal target after {iterations} Grover "
      f"iteration(s) for M={M}, N=2^{N})")


# ---------------------------------------------------------------------------
# 4. Verify: the measured mode(s) must be exactly the classical marked set,
#    and amplitude on marked states must be amplified well above the
#    uniform baseline (M / 2^N).
# ---------------------------------------------------------------------------

top_outcomes_set = set(top_outcomes)
marked_set = set(MARKED_BITSTRINGS)

baseline = M / 2 ** N  # probability of hitting a marked state with no Grover at all
amplification_ok = marked_fraction > 5 * baseline  # should be dramatically higher
modes_match = top_outcomes_set == marked_set

# Independently re-verify that every top outcome truly corresponds to a
# genuine size-6 Sidon subset of the squares (classical re-check, not just
# trusting the earlier brute force).
def bitstring_to_subset(bitstring, squares):
    n = len(squares)
    included = []
    for k, ch in enumerate(bitstring):
        qubit_index = n - 1 - k
        if ch == '1':
            included.append(squares[qubit_index])
    return included


reverified = all(
    len(bitstring_to_subset(bs, SQUARES)) == 6 and is_sidon(bitstring_to_subset(bs, SQUARES))
    for bs in top_outcomes
)

verified = modes_match and amplification_ok and reverified

print(f"\nMeasured top-{M} outcomes match classical marked set: {modes_match}")
print(f"Amplitude amplification above uniform baseline ({baseline:.4f}): "
      f"{amplification_ok} (observed {marked_fraction:.3f})")
print(f"Independent re-verification that top outcomes are genuine size-6 "
      f"Sidon subsets: {reverified}")

if verified:
    print("\nPASS: Grover search on AerSimulator found the largest Sidon "
          f"subset(s) of the first {N} squares, matching classical "
          f"A390813({N}) = {CLASSICAL_BEST_SIZE}.")
else:
    print("\nFAIL: quantum result did not match the classical answer.")

assert verified, "quantum Grover search result did not match classical brute force"
