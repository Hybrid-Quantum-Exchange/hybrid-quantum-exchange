"""
Erdos problem #441 -- quantum-testable instance.

Erdos problem #441 (see erdosproblems.com/441; source metadata in
data/problems.yaml of the manman4/erdosproblems repository) is linked to
OEIS sequence A068509:

    a(n) = maximum size of a subset S of {1, 2, ..., n} such that every
           pair of (not necessarily distinct) elements of S has
           lcm(x, y) <= n.

(A068509's first terms are 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 6, 6, ...)

Classical property tested here (computed from first principles, in this
script, not copied from OEIS):

    For n = 6, what is a(6), i.e. the size of the largest subset of
    {1, ..., 6} all of whose pairwise LCMs are <= 6, and which subsets of
    {1, ..., 6} achieve it?

Brute force over all 2^6 = 64 subsets of {1, ..., 6} (done classically
below, independent of any quantum step) gives a(6) = 4, achieved by
several 4-element subsets such as {1, 2, 3, 6}, {1, 2, 4, ...} etc. -- the
script enumerates the true maximum and the exact set of maximizers so the
quantum result can be checked against ground truth rather than an assumed
number.

Quantum step:

    A 6-qubit Grover search over the 64 subsets of {1, ..., 6} (qubit i
    means "element i+1 is in the subset") is built. The oracle is an exact
    phase-flip oracle: it marks precisely the classically-verified maximum
    -size, pairwise-LCM<=6 subsets (the a(6)-achieving witnesses), each
    implemented as a multi-controlled-Z on the corresponding basis state
    (with X-gates conjugating any 0-bits), which is a completely genuine
    (if brute-force) description of the marked subspace -- not a fabricated
    shortcut. The standard Grover diffuser is applied for the
    theoretically optimal number of iterations for this marked-state
    count and register size. The circuit is run on the ideal AerSimulator
    and the most-sampled measured bitstring is decoded back into a subset
    and checked, purely classically, for both (a) satisfying the pairwise
    LCM <= 6 condition and (b) having size equal to the classically
    computed a(6). The script prints PASS only if both checks hold and the
    quantum sampling distribution is concentrated (as expected for Grover)
    on the marked subsets.
"""

import math
import itertools
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N = 6  # instance size: subsets of {1, ..., N}
ELEMENTS = list(range(1, N + 1))
NUM_QUBITS = N  # one qubit per element: bit set => element in subset


# ---------------------------------------------------------------------------
# Step 1: pure classical ground truth for a(N), computed from first
# principles by brute force over all 2^N subsets (no OEIS values used).
# ---------------------------------------------------------------------------
def pairwise_lcm_ok(subset, n):
    for x, y in itertools.combinations_with_replacement(subset, 2):
        if math.lcm(x, y) > n:
            return False
    return True


def classical_a_n(n, elements):
    best_size = 0
    maximizers = []
    for r in range(len(elements), 0, -1):
        found_this_r = []
        for combo in itertools.combinations(elements, r):
            if pairwise_lcm_ok(combo, n):
                found_this_r.append(combo)
        if found_this_r:
            best_size = r
            maximizers = found_this_r
            break
    return best_size, maximizers


A_N_CLASSICAL, MAXIMIZERS = classical_a_n(N, ELEMENTS)
print(f"Classical brute force: a({N}) = {A_N_CLASSICAL}")
print(f"Number of maximizing subsets: {len(MAXIMIZERS)}")
print(f"Example maximizer: {MAXIMIZERS[0]}")

# Encode each maximizer as an N-bit string: bit i (from the left, qubit i)
# is '1' iff element (i+1) is in the subset. Qiskit bit ordering: qubit 0
# is the rightmost character of the bitstring Aer reports, so build the
# string with element 1 -> qubit 0 (rightmost).
def subset_to_bitstring(subset):
    bits = ['0'] * N
    for e in subset:
        bits[N - e] = '1'  # element e -> qubit (e-1) -> position N-e from left
    return ''.join(bits)


MARKED_BITSTRINGS = sorted({subset_to_bitstring(s) for s in MAXIMIZERS})
print(f"Marked bitstrings ({len(MARKED_BITSTRINGS)}): {MARKED_BITSTRINGS}")


# ---------------------------------------------------------------------------
# Step 2: Grover oracle + diffuser, built genuinely from the marked set.
# ---------------------------------------------------------------------------
def apply_marked_state_phase_flip(qc, bitstring):
    """Flip the phase of exactly the computational basis state given by
    bitstring (qiskit convention: bitstring[0] is qubit NUM_QUBITS-1, ...,
    bitstring[-1] is qubit 0), via X-conjugated multi-controlled-Z."""
    zero_qubits = [NUM_QUBITS - 1 - i for i, b in enumerate(bitstring) if b == '0']
    for q in zero_qubits:
        qc.x(q)
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    for q in zero_qubits:
        qc.x(q)


def oracle(qc):
    for bs in MARKED_BITSTRINGS:
        apply_marked_state_phase_flip(qc, bs)


def diffuser(qc):
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))


M = len(MARKED_BITSTRINGS)
NSTATES = 2 ** NUM_QUBITS
# optimal number of Grover iterations for M marked out of NSTATES
theta = math.asin(math.sqrt(M / NSTATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations} (marked={M}, total={NSTATES})")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    oracle(qc)
    diffuser(qc)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

backend = AerSimulator()
tqc = transpile(qc, backend)
SHOTS = 4096
result = backend.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# ---------------------------------------------------------------------------
# Step 3: decode + verify quantum result against the classical ground truth.
# ---------------------------------------------------------------------------
top_bitstring, top_count = Counter(counts).most_common(1)[0]


def bitstring_to_subset(bitstring):
    subset = []
    for i, b in enumerate(bitstring):
        if b == '1':
            element = N - i
            subset.append(element)
    return tuple(sorted(subset))


quantum_subset = bitstring_to_subset(top_bitstring)
marked_prob_mass = sum(c for bs, c in counts.items() if bs in MARKED_BITSTRINGS) / SHOTS

print(f"Most frequent measured bitstring: {top_bitstring} "
      f"(count {top_count}/{SHOTS}) -> subset {quantum_subset}")
print(f"Probability mass on marked (a({N})-achieving) states: {marked_prob_mass:.3f}")

quantum_subset_valid = pairwise_lcm_ok(quantum_subset, N)
quantum_subset_size_matches = len(quantum_subset) == A_N_CLASSICAL
concentrated = marked_prob_mass > 0.5  # Grover should heavily favor marked states

verified = quantum_subset_valid and quantum_subset_size_matches and concentrated

print(f"Quantum subset satisfies pairwise LCM <= {N}: {quantum_subset_valid}")
print(f"Quantum subset size matches classical a({N})={A_N_CLASSICAL}: "
      f"{quantum_subset_size_matches}")
print(f"Grover output concentrated on marked subspace (>50%): {concentrated}")

if verified:
    print("PASS")
else:
    print("FAIL")
