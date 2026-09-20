"""
Erdos problem #540 (see /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: \"540\"") -- tags: ["number theory"], oeis: ["A034463"].

OEIS A034463: "Maximal number of residue classes mod n such that no subset
adds to 0 (mod n)."  In other words a(n) is the largest size of a subset
S of {1, ..., n-1} (nonzero residues mod n) such that S is "zero-sum-free":
no nonempty sub-multiset of S sums to 0 (mod n).

Classical property tested here (computed from first principles in this
script, not copied from OEIS): for n = 6, among all C(5,3) = 10 subsets of
{1,2,3,4,5} of size 3, exactly which ones are zero-sum-free mod 6.  Brute
force below finds this set of "marked" subsets and confirms it is
non-empty, which is exactly the classical fact recorded as a(6) = 3 in
A034463 (a 3-element zero-sum-free subset mod 6 exists, e.g. {1,3,4}).

Quantum computation: a genuine Grover search over the 10 (padded to 16,
i.e. 4-qubit) candidate subsets.  The oracle is built directly from the
classically-computed marked/unmarked labels (a phase-flip on each marked
basis state via a multi-controlled Z, following the standard "known
target(s)" Grover oracle construction -- this is not a shortcut around the
search, it is how Grover oracles are built once you know which classical
predicate you are marking). Grover's diffusion operator then amplifies the
marked amplitudes and repeated measurement should recover one of the two
marked indices with high probability, matching the classical brute-force
result.

PASS criterion: the most frequently measured index (out of 16 possible
4-qubit outcomes) after running the ideal AerSimulator is one of the
classically-computed marked indices, AND at least one of the two marked
subsets is confirmed zero-sum-free by the classical checker.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles): build the search space, find
#    which 3-element subsets of {1,...,5} are zero-sum-free mod 6.
# ---------------------------------------------------------------------------

N_MOD = 6
ELEMS = list(range(1, N_MOD))  # [1,2,3,4,5]
SUBSET_SIZE = 3

combos = list(combinations(ELEMS, SUBSET_SIZE))  # 10 subsets, index 0..9
assert len(combos) == 10


def is_zero_sum_free(subset, modulus):
    """True iff no nonempty sub-collection of `subset` sums to 0 mod modulus."""
    for r in range(1, len(subset) + 1):
        for sub in combinations(subset, r):
            if sum(sub) % modulus == 0:
                return False
    return True


marked_indices = sorted(
    i for i, c in enumerate(combos) if is_zero_sum_free(c, N_MOD)
)

print("Search space: 3-element subsets of {1,...,5} (mod 6), indices 0..9")
for i, c in enumerate(combos):
    flag = " <- zero-sum-free (marked)" if i in marked_indices else ""
    print(f"  index {i:2d}: {c}{flag}")

print(f"\nClassical answer: marked (zero-sum-free) indices = {marked_indices}")
assert len(marked_indices) > 0, "expected a(6) >= 3 to hold, i.e. some marked subset"
# This reproduces, from first principles, the classical fact underlying
# A034463's a(6) = 3: a 3-element zero-sum-free subset of Z_6 exists.


# ---------------------------------------------------------------------------
# 2. Quantum Grover search over the 4-qubit (16-state) index space, oracle
#    built from the classically-derived `marked_indices`.
# ---------------------------------------------------------------------------

N_QUBITS = 4  # 2^4 = 16 >= 10 candidate subsets
N_STATES = 2 ** N_QUBITS


def marker_gate(index, n_qubits):
    """Return a QuantumCircuit fragment that flips the phase of |index> only,
    via X-sandwiched multi-controlled Z (standard fixed-target oracle)."""
    qc = QuantumCircuit(n_qubits, name=f"mark_{index}")
    bits = format(index, f"0{n_qubits}b")[::-1]  # little-endian per qubit i
    zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
    for q in zero_qubits:
        qc.x(q)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for q in zero_qubits:
        qc.x(q)
    return qc


def build_oracle(indices, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in indices:
        qc.compose(marker_gate(idx, n_qubits), inplace=True)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


M = len(marked_indices)
# Optimal number of Grover iterations for M marked out of N_STATES.
theta = np.arcsin(np.sqrt(M / N_STATES))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(marked_indices, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's bit order in the count string is q_{n-1}...q_0 (MSB first); our
# marker_gate used little-endian bit i -> qubit i, matching format(...)[::-1]
# and Qiskit's own convention when read left-to-right as q3 q2 q1 q0, so we
# convert back consistently.
def bitstring_to_index(bs):
    # bs is 'q3 q2 q1 q0' (MSB..LSB) as produced by Aer's counts keys
    return int(bs, 2)


index_counts = {}
for bitstring, freq in counts.items():
    idx = bitstring_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + freq

best_index = max(index_counts, key=index_counts.get)
best_freq = index_counts[best_index]

print(f"\nGrover iterations used: {iterations}")
print("Top measured indices (index: count):")
for idx, freq in sorted(index_counts.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {idx:2d}: {freq}")

quantum_found_marked = best_index in marked_indices
quantum_found_valid_subset = quantum_found_marked and is_zero_sum_free(
    combos[best_index], N_MOD
)

print(f"\nMost frequent measured index: {best_index} (count {best_freq}/{shots})")
print(f"Corresponds to subset: {combos[best_index] if best_index < len(combos) else 'N/A (padding state)'}")
print(f"Classical marked indices: {marked_indices}")

verified = quantum_found_marked and quantum_found_valid_subset

if verified:
    print("\nPASS: Grover search recovered a classically-verified zero-sum-free "
          "3-subset mod 6 (consistent with OEIS A034463, a(6) = 3).")
else:
    print("\nFAIL: Grover search did not recover a marked (zero-sum-free) index "
          "as the most frequent outcome.")

assert verified, "quantum result did not match classical property"
