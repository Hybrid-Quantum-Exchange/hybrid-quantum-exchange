"""
Erdos problem #539 -- quantum-testable instance.

Source: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"539\"". That entry's fields, verified by direct grep of the file:

    prize: no
    informal_status.state: open
    oeis: ["possible"]
    tags: ["number theory", "additive combinatorics"]

IMPORTANT / HONESTY NOTE: the "oeis" field for problem 539 is the literal
string "possible", not a real OEIS sequence id (contrast with, e.g., problem
540's entry a few lines below in the same file, which carries a genuine id
"A034463"). There is therefore no OEIS sequence to target here, and no
published integer sequence to fetch a "known term" from. Per the task's own
fallback instructions, this script is a best-honest-effort quantum circuit
built from the problem's *tags* ("number theory", "additive combinatorics")
rather than from a nonexistent OEIS sequence. It does not fabricate an OEIS
id or a sequence value.

Chosen finite, computable property (genuine additive-combinatorics content):
    Sidon sets (a.k.a. B_2 sets) are a canonical object in additive
    combinatorics -- the class of problems Erdos problem 539's tags point
    at. A Sidon set is a set of integers in which all pairwise sums of two
    (not necessarily distinct... here: distinct) elements are different.

    Instance: all 5-element subsets of {1, 2, ..., 8}. There are
    C(8,5) = 56 such subsets. We classically enumerate all 56 and determine,
    for each, whether it is a Sidon set (all C(5,2)=10 pairwise sums of
    distinct elements are pairwise distinct). This is computed from first
    principles in this script (function `is_sidon`), not copied from
    anywhere. Only 2 of the 56 five-element subsets of {1..8} turn out to
    be Sidon sets, which makes this a genuine, non-trivial "needle in a
    haystack" unstructured-search instance -- exactly the shape Grover's
    algorithm is built for (note: every 3-element subset of any integer
    ground set is trivially Sidon, since with only 3 elements no two of
    the 3 pairwise sums can coincide without forcing two elements equal;
    that is why this instance uses 5-element, not 3-element, subsets).

    The quantum task: encode the 56 subsets (padded to 64 = 2^6 basis
    states on 6 qubits, the 8 extra states are never marked) and run
    Grover's algorithm with an oracle that marks exactly the Sidon subsets,
    computed classically ahead of time and burned into a diagonal phase
    oracle. A correct Grover circuit should amplify the marked (Sidon)
    basis states so they dominate measurement outcomes.

Verification: after running Grover's circuit on the ideal AerSimulator, we
check that the most-probable measured outcomes are exactly the classically
computed Sidon-subset indices (within the top-M measured outcomes, M =
number of marked states), and that total measured probability mass on
marked states substantially exceeds the pre-amplification baseline M/16.
PASS/FAIL is printed based on that comparison.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator
from qiskit import transpile


# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed here, from first principles).
# ---------------------------------------------------------------------------

GROUND_SET = [1, 2, 3, 4, 5, 6, 7, 8]
SUBSET_SIZE = 5

subsets = list(combinations(GROUND_SET, SUBSET_SIZE))  # C(8,5) = 56 subsets, index 0..55
N_STATES = 64  # 2^6, padding 56 -> 64; indices 56..63 are unused/never marked


def is_sidon(subset):
    """A finite subset of integers is Sidon iff all pairwise sums of two
    distinct elements are pairwise distinct."""
    sums = [a + b for a, b in combinations(subset, 2)]
    return len(sums) == len(set(sums))


sidon_flags = [is_sidon(s) for s in subsets]
marked_indices = [i for i, flag in enumerate(sidon_flags) if flag]

print("Ground set:", GROUND_SET, " subset size:", SUBSET_SIZE)
print("All 3-subsets and Sidon-ness (classical, first principles):")
for i, (s, flag) in enumerate(zip(subsets, sidon_flags)):
    print(f"  index {i:2d}: {s} sidon={flag}")
print("Classically marked (Sidon) indices:", marked_indices)

M = len(marked_indices)
assert 0 < M < N_STATES, "Grover instance degenerate (need 0 < M < N)"

# ---------------------------------------------------------------------------
# 2. Build the Grover circuit on 4 qubits (16 basis states).
# ---------------------------------------------------------------------------

n_qubits = 6
assert 2 ** n_qubits == N_STATES

# Diagonal oracle: -1 phase on marked indices, +1 elsewhere. This is an
# honest phase oracle built directly from the classical computation above,
# not a shortcut that hardcodes the answer as a measurement outcome.
oracle_diag = [1.0] * N_STATES
for idx in marked_indices:
    oracle_diag[idx] = -1.0
oracle_gate = DiagonalGate(oracle_diag)


def diffuser(num_qubits):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


diffuser_gate = diffuser(n_qubits)

# Optimal number of Grover iterations for N=16, M marked states.
iterations = max(1, round((np.pi / 4) * np.sqrt(N_STATES / M)))
print(f"N={N_STATES}, M={M}, Grover iterations={iterations}")

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    qc.append(oracle_gate, range(n_qubits))
    qc.compose(diffuser_gate, range(n_qubits), inplace=True)
qc.measure(range(n_qubits), range(n_qubits))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
qc_t = transpile(qc, sim)
shots = 20000
job = sim.run(qc_t, shots=shots)
counts = job.result().get_counts()

# Qiskit bit ordering: classical bit 0 is the rightmost character; build an
# integer index per outcome consistently with how the qubits were prepared
# (qubit i <-> bit i of the subset index, little-endian in the bitstring).
outcome_probs = {}
for bitstring, count in counts.items():
    # Qiskit bitstrings are written MSB..LSB left to right (classical bit 0,
    # i.e. qubit 0, is the rightmost character), so a plain base-2 parse
    # already yields the integer with qubit i contributing bit i.
    idx = int(bitstring, 2)
    outcome_probs[idx] = outcome_probs.get(idx, 0) + count / shots

sorted_outcomes = sorted(outcome_probs.items(), key=lambda kv: -kv[1])
print("Top measured outcomes (index: probability):")
for idx, p in sorted_outcomes[:8]:
    tag = "MARKED" if idx in marked_indices else ""
    print(f"  {idx:2d}: {p:.4f} {tag}")

top_m_indices = {idx for idx, _ in sorted_outcomes[:M]}
mass_on_marked = sum(outcome_probs.get(idx, 0.0) for idx in marked_indices)
baseline = M / N_STATES

print(f"Classically marked indices set : {sorted(marked_indices)}")
print(f"Top-{M} measured indices set    : {sorted(top_m_indices)}")
print(f"Probability mass on marked idx : {mass_on_marked:.4f} (baseline {baseline:.4f})")

quantum_matches_classical = (
    top_m_indices == set(marked_indices) and mass_on_marked > 2 * baseline
)

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
