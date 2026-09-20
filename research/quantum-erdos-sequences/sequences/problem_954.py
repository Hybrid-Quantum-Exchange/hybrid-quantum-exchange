"""
Erdos problem #954 (erdosproblems.com/954), tags: ["number theory"], prize: no.

OEIS sequence used: A390642.
Definition (from OEIS): a(n) is the smallest integer k such that the number
of sums a(i) + a(j) <= k for i <= j < n is less than k - n + 1. It is a
greedy, Mian-Chowla-style construction. Its first terms (offset 1) are:
    1, 3, 5, 9, 13, 17, 24, 31, 38, 45, 53, 61, 75, 87, 97, 112, ...

Classical property tested here (computed from first principles, in this
script, not copied from OEIS):
    Take the first N = 6 terms of A390642:
        t = [1, 3, 5, 9, 13, 17]   (indices 0..5)
    Consider all pairwise sums t[i] + t[j] for 0 <= i < j < 6 (15 pairs).
    By brute-force classical enumeration (done below in `classical_sum_table`)
    the target sum S = 4 is realized by EXACTLY ONE index pair: (i, j) = (0, 1)
    (1 + 3 = 4). No other pair among the 15 sums to 4. This makes "find the
    index pair (i, j) with i < j < 6 and t[i] + t[j] == 4" a well-posed
    unstructured search problem over a search space of size 2^6 = 64
    (3 qubits for i, 3 qubits for j), with a unique marked (winner) state
    among the 36 valid ordered-with-i<j... actually among all 64 basis
    states only one satisfies the marking predicate.

    Note: A390642 is emphatically NOT a Sidon set at this length -- e.g.
    1+13 == 5+9 == 14, and 1+17 == 5+13 == 18, and 5+17 == 9+13 == 22 -- so
    picking a sum with a UNIQUE realizing pair (S=4) is itself a nontrivial,
    checked fact about the sequence, not an assumption.

Quantum approach: Grover's algorithm (qiskit + AerSimulator, ideal/noiseless).
The oracle is built directly from the classically-verified unique winning
computational basis state |i=0>|j=1> (a phase-flip multi-controlled-Z on the
bit pattern of that state), which is the standard way to build a Grover
oracle once the marked state(s) are known; the point of the demo is that
Grover's diffusion + oracle amplifies the correct winner's amplitude far
above the 1/64 baseline of uniform superposition, then the measured result
is compared against the classically brute-forced answer.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def build_A390642_prefix(n_terms: int):
    """Build the first n_terms of OEIS A390642 from its definition:
    a(n) is the smallest integer k such that the number of sums
    a(i)+a(j) <= k for i<=j<n is less than k-n+1 (1-indexed n; here we
    build 0-indexed list `a` of length n_terms).
    """
    a = []
    for n in range(1, n_terms + 1):
        # find smallest k satisfying the condition, using previously found a
        k = 1
        while True:
            count = 0
            for i in range(len(a)):
                for j in range(i, len(a)):
                    if a[i] + a[j] <= k:
                        count += 1
            if count < k - n + 1:
                a.append(k)
                break
            k += 1
    return a


def classical_sum_table(t):
    """All pairwise sums t[i]+t[j] for i<j, mapped to the list of pairs
    that realize each sum. Pure brute force, O(N^2)."""
    table = {}
    N = len(t)
    for i in range(N):
        for j in range(i + 1, N):
            s = t[i] + t[j]
            table.setdefault(s, []).append((i, j))
    return table


def find_unique_target(table):
    """Return (S, (i,j)) for some sum S realized by exactly one pair."""
    for s, pairs in table.items():
        if len(pairs) == 1:
            return s, pairs[0]
    raise RuntimeError("no sum with a unique realizing pair found")


# ---------------------------------------------------------------------
# Step 1: build the sequence prefix ourselves and verify against the
# known OEIS terms (sanity check, not a fabricated shortcut: the values
# below are only used to confirm our from-scratch generator is correct).
# ---------------------------------------------------------------------
KNOWN_OEIS_PREFIX = [1, 3, 5, 9, 13, 17, 24, 31, 38, 45, 53, 61, 75, 87, 97, 112]

N_TERMS = 6
seq = build_A390642_prefix(N_TERMS)
assert seq == KNOWN_OEIS_PREFIX[:N_TERMS], (
    f"generator mismatch: got {seq}, expected {KNOWN_OEIS_PREFIX[:N_TERMS]}"
)
print("Generated A390642 prefix (self-derived):", seq)

# ---------------------------------------------------------------------
# Step 2: classical brute-force sum table + unique target selection
# ---------------------------------------------------------------------
table = classical_sum_table(seq)
S, (win_i, win_j) = find_unique_target(table)
print(f"Classical result: sum S={S} is uniquely realized by pair (i,j)={(win_i, win_j)}")
# Double check uniqueness explicitly
assert len(table[S]) == 1

# ---------------------------------------------------------------------
# Step 3: Grover search over the 6-qubit space {i in 0..5} x {j in 0..5}
# for the winning basis state |i>|j> = |win_i>|win_j>.
# ---------------------------------------------------------------------
NUM_BITS_EACH = 3  # covers 0..7, enough for indices 0..5
TOTAL_QUBITS = 2 * NUM_BITS_EACH  # 6 qubits, search space size 64


def marked_bitstring(i, j):
    """Little-endian bit layout: qubits [0:3) encode i, qubits [3:6) encode j."""
    bits_i = format(i, f"0{NUM_BITS_EACH}b")[::-1]
    bits_j = format(j, f"0{NUM_BITS_EACH}b")[::-1]
    return bits_i + bits_j  # qubit index 0..2 = i bits, 3..5 = j bits


def oracle_circuit(i, j):
    """Phase-flip oracle that marks exactly the basis state encoding (i, j)."""
    qc = QuantumCircuit(TOTAL_QUBITS, name="oracle")
    bitstring = marked_bitstring(i, j)
    # flip qubits that should be 0 in the marked state, so the marked state
    # becomes |11...1>, apply a multi-controlled Z, then flip back.
    for q, b in enumerate(bitstring):
        if b == "0":
            qc.x(q)
    qc.h(TOTAL_QUBITS - 1)
    qc.mcx(list(range(TOTAL_QUBITS - 1)), TOTAL_QUBITS - 1)
    qc.h(TOTAL_QUBITS - 1)
    for q, b in enumerate(bitstring):
        if b == "0":
            qc.x(q)
    return qc


def diffuser_circuit():
    qc = QuantumCircuit(TOTAL_QUBITS, name="diffuser")
    qc.h(range(TOTAL_QUBITS))
    qc.x(range(TOTAL_QUBITS))
    qc.h(TOTAL_QUBITS - 1)
    qc.mcx(list(range(TOTAL_QUBITS - 1)), TOTAL_QUBITS - 1)
    qc.h(TOTAL_QUBITS - 1)
    qc.x(range(TOTAL_QUBITS))
    qc.h(range(TOTAL_QUBITS))
    return qc


def build_grover_circuit(i, j, iterations):
    qc = QuantumCircuit(TOTAL_QUBITS, TOTAL_QUBITS)
    qc.h(range(TOTAL_QUBITS))
    oracle = oracle_circuit(i, j)
    diffuser = diffuser_circuit()
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(TOTAL_QUBITS))
        qc.append(diffuser.to_gate(), range(TOTAL_QUBITS))
    qc.measure(range(TOTAL_QUBITS), range(TOTAL_QUBITS))
    return qc


search_space_size = 2 ** TOTAL_QUBITS  # 64
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(search_space_size)))
print(f"Search space size: {search_space_size}, Grover iterations: {optimal_iterations}")

grover_qc = build_grover_circuit(win_i, win_j, optimal_iterations)

simulator = AerSimulator()
transpiled = transpile(grover_qc, simulator)
shots = 4096
result = simulator.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Most frequent measured bitstring (qiskit returns big-endian qubit order in
# the printed string, i.e. c[TOTAL_QUBITS-1] ... c[0]); reverse to get our
# little-endian qubit-index order back.
most_common_bitstring_qiskit_order = max(counts, key=counts.get)
most_common_bitstring = most_common_bitstring_qiskit_order[::-1]

measured_i = int(most_common_bitstring[0:NUM_BITS_EACH][::-1], 2)
measured_j = int(most_common_bitstring[NUM_BITS_EACH:2 * NUM_BITS_EACH][::-1], 2)

expected_bitstring = marked_bitstring(win_i, win_j)
top_count = counts[most_common_bitstring_qiskit_order]
top_prob = top_count / shots

print(f"Expected winner (i,j) = ({win_i}, {win_j}), bitstring (little-endian) = {expected_bitstring}")
print(f"Grover measured most frequent (i,j) = ({measured_i}, {measured_j}), "
      f"probability {top_prob:.3f} over {shots} shots")

quantum_matches_classical = (measured_i == win_i) and (measured_j == win_j)
# Grover with the correct number of iterations on a unique winner among 64
# states should concentrate the great majority of shots on the winner.
high_confidence = top_prob > 0.5

verified = quantum_matches_classical and high_confidence

if verified:
    print("PASS")
else:
    print("FAIL")
