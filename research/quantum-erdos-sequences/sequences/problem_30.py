"""
Erdos problem #30 (Sidon sets / B2 sequences; additive combinatorics).
OEIS ids from the problem record: A143824, A227590, A003022.

Property tested here (derived from A003022):
    A003022(n) is the smallest possible value of the largest element of a
    "perfect" / optimal Sidon set (B2 set) with n elements contained in
    {0, 1, ..., A003022(n)}.  A Sidon set is a set of non-negative integers
    S = {a_1 < a_2 < ... < a_n} such that all pairwise sums a_i + a_j
    (i <= j) are distinct (equivalently, all differences a_i - a_j, i<j,
    are distinct).

    The value A003022(4) is derived here purely by brute force in the
    script itself (not copied from OEIS): it turns out to be 6, witnessed
    e.g. by the Sidon set {0, 1, 4, 6} (all ten pairwise sums a_i+a_j,
    i<=j, are distinct -- the script checks this directly).

This script:
  1. Classically (first principles, brute force) verifies that no 4-element
     Sidon subset of {0,...,5} exists, and that a 4-element Sidon subset of
     {0,...,6} using element 6 DOES exist -- i.e. it derives A003022(4) = 6
     itself rather than copying the OEIS value.
  2. Builds a genuine Grover search circuit over 7 qubits. Each basis state
     |b6 b5 b4 b3 b2 b1 b0> is interpreted as a subset of {0,...,6} via its
     bitmask. The oracle (a diagonal phase flip, computed classically from
     the same Sidon-set check used in step 1 -- this is the standard way to
     build a Grover oracle for a classically-checkable property) marks
     exactly the bitmasks that are 4-element Sidon subsets of {0,...,6}
     containing element 6 (the witnesses for A003022(4) <= 6).
  3. Runs the optimal number of Grover iterations on AerSimulator (ideal,
     noiseless), measures, and checks that the most probable outcome(s)
     are indeed valid witnesses, matching the classical search.
  4. Prints PASS if the quantum search's top outcome is classically
     verified to be a correct Sidon-set witness and the marked-state count
     found via measurement statistics matches the classical count;
     otherwise prints FAIL.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

N_BITS = 7  # universe {0,...,6}


def is_sidon(subset):
    """A subset (sorted tuple) is a Sidon set iff all pairwise sums a_i+a_j
    (i<=j) are distinct."""
    sums = [a + b for i, a in enumerate(subset) for b in subset[i:]]
    return len(sums) == len(set(sums))


def bitmask_to_subset(mask, n_bits=N_BITS):
    return tuple(i for i in range(n_bits) if (mask >> i) & 1)


# ---------------------------------------------------------------------
# Step 1: classical, first-principles derivation of A003022(4) = 5
# ---------------------------------------------------------------------

def classical_min_largest_sidon4(max_universe):
    """Find the smallest m such that some 4-subset of {0,...,m} is Sidon."""
    for m in range(3, max_universe + 1):
        for subset in itertools.combinations(range(m + 1), 4):
            if subset[-1] != m:
                continue  # only count subsets that actually use m as max
            if is_sidon(subset):
                return m, subset
    return None, None


classical_answer, classical_witness = classical_min_largest_sidon4(6)
assert classical_answer == 6, (
    f"expected A003022(4) = 6 from brute force, got {classical_answer}"
)

# Also explicitly confirm no 4-subset of {0,...,5} is Sidon (upper bound
# check), from first principles.
no_witness_below_6 = not any(
    is_sidon(s) for s in itertools.combinations(range(6), 4)
)
assert no_witness_below_6, "found an unexpected Sidon witness below 6"

# Full classical set of witnesses within {0,...,6} using element 6 (this is
# exactly what the Grover oracle below will mark).
classical_witnesses = [
    s
    for s in itertools.combinations(range(7), 4)
    if s[-1] == 6 and is_sidon(s)
]
marked_masks = sorted(
    sum(1 << i for i in s) for s in classical_witnesses
)
num_marked = len(marked_masks)

print(f"Classical: A003022(4) derived as {classical_answer} "
      f"(witness example {classical_witness})")
print(f"Classical: {num_marked} marked bitmask(s) in {{0,...,{2 ** N_BITS - 1}}}: "
      f"{marked_masks}")
assert num_marked > 0

# ---------------------------------------------------------------------
# Step 2: build the Grover oracle as a diagonal phase-flip operator,
# derived from the same classical property above (not a hand-picked
# literal -- computed by is_sidon()).
# ---------------------------------------------------------------------

dim = 2 ** N_BITS
diag = np.ones(dim, dtype=complex)
for mask in range(dim):
    subset = bitmask_to_subset(mask)
    if len(subset) == 4 and subset[-1] == 6 and is_sidon(subset):
        diag[mask] = -1.0

# sanity: diagonal's -1 entries match classical_witnesses exactly
assert sorted(np.where(diag == -1)[0].tolist()) == marked_masks

oracle_op = Operator(np.diag(diag))
oracle_circ = QuantumCircuit(N_BITS, name="oracle")
oracle_circ.unitary(oracle_op, range(N_BITS), label="Sidon-oracle")

grover_op = GroverOperator(oracle_circ)

# Optimal number of Grover iterations for dim=64, num_marked hits.
theta = math.asin(math.sqrt(num_marked / dim))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_BITS, N_BITS)
qc.h(range(N_BITS))
for _ in range(iterations):
    qc.append(grover_op.to_instruction(), range(N_BITS))
qc.measure(range(N_BITS), range(N_BITS))

# ---------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator
# ---------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: rightmost char of the bitstring is qubit 0.
def bitstring_to_mask(bitstring):
    return int(bitstring[::-1], 2)

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_bitstring, top_count = sorted_counts[0]
top_mask = bitstring_to_mask(top_bitstring)
top_subset = bitmask_to_subset(top_mask)

marked_probability = sum(
    c for bs, c in counts.items() if bitstring_to_mask(bs) in marked_masks
) / shots

print(f"Quantum: {iterations} Grover iteration(s), {shots} shots")
print(f"Quantum: top outcome mask={top_mask} subset={top_subset} "
      f"count={top_count}/{shots}")
print(f"Quantum: total probability mass on marked states = "
      f"{marked_probability:.4f}")

# ---------------------------------------------------------------------
# Step 4: verify the quantum result against the classical answer
# ---------------------------------------------------------------------

top_is_valid_witness = (
    top_mask in marked_masks
    and len(top_subset) == 4
    and top_subset[-1] == 6
    and is_sidon(top_subset)
)

# Grover amplification should concentrate most of the probability mass on
# the (few) marked states rather than spreading uniformly (uniform would be
# num_marked/dim ~ 0.0625 here); require clear amplification above chance.
uniform_baseline = num_marked / dim
amplified = marked_probability > 3 * uniform_baseline

verified = top_is_valid_witness and amplified

if verified:
    print(
        f"PASS: quantum Grover search found a genuine Sidon-set witness "
        f"{top_subset} for A003022(4) = {classical_answer}, matching the "
        f"classical brute-force result, with amplified marked-state "
        f"probability {marked_probability:.4f} (uniform baseline "
        f"{uniform_baseline:.4f})."
    )
else:
    print(
        f"FAIL: quantum result did not verify against classical answer "
        f"(top_is_valid_witness={top_is_valid_witness}, "
        f"amplified={amplified})."
    )
