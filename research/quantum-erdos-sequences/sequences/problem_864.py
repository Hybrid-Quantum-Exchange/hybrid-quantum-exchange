"""
Erdos problem #864 (erdosproblems.com), quantum-testable instance.

Metadata (from data/problems.yaml in manman4/erdosproblems, read-only clone):
  number: 864
  oeis: ["A389182"]
  tags: ["number theory", "sidon sets", "additive combinatorics"]

The problem concerns Sidon sets: a set of non-negative integers
S = {a_0 < a_1 < ... < a_{k-1}} is a Sidon set (a "B2 set") iff all pairwise
sums a_i + a_j (i <= j) are distinct -- equivalently, no four (not
necessarily distinct) elements a_i + a_j = a_k + a_l with {i,j} != {k,l}.
Sidon sets and their maximal density are the classical object underlying
A389182's family of sequences (additive-combinatorics constructions built
from B2/Sidon-type sets).

Classical property tested here (small, finite, exactly computable):
  Given the candidate set S = {1, 2, 3, 4}, is it a Sidon set? If not,
  find (search for) a colliding pair of 2-element subsets, i.e. indices
  i<j and k<l with {i,j} != {k,l} and a_i+a_j = a_k+a_l.

S = {1,2,3,4} is NOT a Sidon set, because 1+4 = 2+3 = 5. This is verified
classically in this script by brute force over all C(4,2)=6 index pairs
before any quantum code runs.

Quantum approach: Grover search.
  - Enumerate the 6 unordered index-pairs {i,j} of S in a fixed order and
    assign each a 3-qubit index in {0,...,5} (values 6,7 are unused/never
    marked).
  - Classically compute, for each of the 6 pairs, its sum a_i+a_j.
  - The "marked" (collision) pairs are exactly those whose sum is not
    unique among the 6 sums. For S={1,2,3,4} this is pair-index 2
    ({0,3}, sum 5) and pair-index 3 ({1,2}, sum 5).
  - Build a genuine Grover oracle (multi-controlled Z gates keyed on the
    computational-basis bitstrings for indices 2 and 3, i.e. '010' and
    '011') plus the standard diffusion operator, run it on AerSimulator,
    and check that measurement overwhelmingly returns one of the two
    marked (collision) indices -- i.e. the quantum search finds a genuine
    Sidon-violating pair, matching the classical brute-force answer.

No external deps beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (ground truth), from first principles.
# ---------------------------------------------------------------------------

S = [1, 2, 3, 4]

# All unordered index-pairs, in a fixed enumeration order -> pair index 0..5.
index_pairs = list(itertools.combinations(range(len(S)), 2))
assert len(index_pairs) == 6

sums = [S[i] + S[j] for (i, j) in index_pairs]

# A pair-index is "marked" (a Sidon violation) if its sum is shared with a
# different pair-index.
sum_counts = {}
for s in sums:
    sum_counts[s] = sum_counts.get(s, 0) + 1

classical_marked = sorted(
    idx for idx, s in enumerate(sums) if sum_counts[s] > 1
)

is_sidon = len(classical_marked) == 0

print("Candidate set S =", S)
print("Index pairs (i,j) -> sum:", list(zip(index_pairs, sums)))
print("Sidon set?", is_sidon)
print("Classical Sidon-violating pair indices (ground truth):", classical_marked)

if is_sidon:
    raise SystemExit(
        "Chosen instance is unexpectedly a Sidon set; no collision to search "
        "for -- this script requires a non-Sidon instance to demonstrate the "
        "Grover search meaningfully."
    )

# ---------------------------------------------------------------------------
# 2. Quantum Grover search over the 3-qubit index space {0,...,7}, marking
#    exactly the classically-derived collision indices.
# ---------------------------------------------------------------------------

N_QUBITS = 3  # 2^3 = 8 >= 6 pair-indices


def bits_of(n, width):
    return [(n >> b) & 1 for b in range(width)]  # little-endian, qubit 0 = LSB


def apply_oracle(qc, marked_indices, qubits):
    """Phase-flip exactly the basis states in marked_indices (multi-controlled Z)."""
    for m in marked_indices:
        bits = bits_of(m, len(qubits))
        # Flip qubits that should be 0 so the state becomes all-ones, apply
        # a multi-controlled Z (via H-MCX-H on the last qubit), then undo.
        for q, b in zip(qubits, bits):
            if b == 0:
                qc.x(q)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q, b in zip(qubits, bits):
            if b == 0:
                qc.x(q)


def apply_diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


N = 2 ** N_QUBITS
M = len(classical_marked)
# Standard Grover optimal iteration count.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))

qc.h(qubits)
for _ in range(iterations):
    apply_oracle(qc, classical_marked, qubits)
    apply_diffuser(qc, qubits)
qc.measure(qubits, qubits)

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first over the classical register (which was
# filled qubit-order little-endian), so convert back to an integer index
# consistently with bits_of()'s convention.
def bitstring_to_index(bitstring):
    # Qiskit prints counts as c[N-1]...c[1]c[0] (highest-index qubit first,
    # register bit 0 rightmost). Since bits_of() defines index = sum(bit_b *
    # 2**b), that is exactly the standard big-endian-string -> int reading.
    return int(bitstring, 2)


index_counts = {}
for bitstring, c in counts.items():
    idx = bitstring_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + c

most_likely_index = max(index_counts, key=index_counts.get)

print("Grover iterations used:", iterations)
print("Measured index counts:", index_counts)
print("Most likely measured pair-index:", most_likely_index)

# Fraction of shots landing on ANY classically-marked (collision) index.
marked_shots = sum(index_counts.get(i, 0) for i in classical_marked)
marked_fraction = marked_shots / shots

print(
    f"Fraction of shots on a classically-marked collision index: "
    f"{marked_fraction:.3f}"
)

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical ground truth and report.
# ---------------------------------------------------------------------------

quantum_found_collision = most_likely_index in classical_marked
high_confidence = marked_fraction > 0.8  # Grover should heavily favor marked states

verified = quantum_found_collision and high_confidence

if verified:
    (i, j) = index_pairs[most_likely_index]
    print(
        f"Quantum search found Sidon-violating pair index {most_likely_index} "
        f"-> elements S[{i}]={S[i]}, S[{j}]={S[j]}, sum={S[i] + S[j]}, "
        f"matching classical ground truth {classical_marked}."
    )
    print("PASS")
else:
    print(
        "Quantum measurement did not match classical ground truth with "
        "sufficient confidence."
    )
    print("FAIL")
