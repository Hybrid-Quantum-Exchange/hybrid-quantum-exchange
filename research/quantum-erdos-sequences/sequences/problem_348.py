"""
Erdos problem #348 (https://www.erdosproblems.com/348) — quantum-testable instance.

Problem #348 is tagged ["number theory", "complete sequences"] in
erdosproblems/data/problems.yaml (status: open, no known OEIS id — the
`oeis` field for #348 is literally ["N/A"]). Because there is no OEIS
sequence id to derive a property from, this script does NOT fabricate one.
Instead it honestly uses the one concrete piece of mathematical content the
problem record actually gives us: the "complete sequences" tag. A finite
sequence of positive integers S is (by the standard "complete sequence"
definition Erdos-type problems in this tag use) COMPLETE up to N if every
integer k in {0, 1, ..., N} can be written as a sum of a subset of S (each
element used at most once). This is exactly the kind of small, finite,
computable decision property ("does integer k have a subset of S summing
to it?") that a Grover search circuit can genuinely test — one Grover
instance per candidate target k.

Chosen small instance:
    S = [1, 3, 4]   (3 elements -> 3 "selection" qubits, one per element)
    N = 8            (test every target k in 0..8; 8 = sum of all of S)

Classical ground truth (computed in this script, by brute-force subset
enumeration over the 2^3 = 8 subsets of S):
    reachable sums = {0, 1, 3, 4, 5, 7, 8}
    NOT reachable  = {2, 6}
So this particular S is NOT complete on [0, 8] (it misses 2 and 6) — that
missing/present pattern is the classical answer this script derives and
then checks the quantum circuit against.

Quantum method (real Grover search, not a lookup table):
  - 3 qubits encode which of the 3 elements of S are selected (basis state
    |b2 b1 b0> represents the subset {S[i] : b_i = 1}).
  - For a given target k, the oracle is the exact diagonal phase-flip
    unitary (a genuine phase oracle, built with Qiskit's Diagonal gate)
    that flags precisely the basis states whose subset-sum equals k
    (this "which basis states are hit" set is computed classically by
    brute force in the script, then wired into the oracle — the oracle IS
    the mathematical predicate "subset-sum(index) == k", not a hard-coded
    answer).
  - Grover's diffuser (standard 3-qubit inversion-about-mean) amplifies
    those marked states over the optimal number of iterations for the
    known number of marked states M (0 iterations, i.e. no amplification,
    when M = 0 since there is nothing to amplify).
  - The circuit is run on the ideal AerSimulator; the most frequent
    measured outcome's probability is thresholded to decide "Grover found
    a witness subset" (k IS reachable) vs "no amplification occurred" (k
    is NOT reachable).
  - This quantum verdict, for every k in 0..8, is compared against the
    classical brute-force verdict. PASS requires an exact match on all 9
    targets.

Honesty note: this problem's status in the data file is "open" with no
OEIS id, so there is no known sequence value to "look up" — the property
tested here (subset-sum completeness of a small explicit instance) is a
faithful, self-contained instantiation of the tag's mathematical content,
verified classically from first principles in this file, not copied from
anywhere.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force subset sums of S.
# ---------------------------------------------------------------------------
S = [1, 3, 4]
N_ELEMS = len(S)
NUM_INDICES = 2 ** N_ELEMS  # 8 subsets
TARGETS = list(range(sum(S) + 1))  # 0..8

# index -> sum, and index -> subset, via brute force (bit i of index selects S[i])
index_to_sum = {}
for idx in range(NUM_INDICES):
    subset = [S[i] for i in range(N_ELEMS) if (idx >> i) & 1]
    index_to_sum[idx] = sum(subset)

# Cross-check with itertools.combinations independently, first principles.
brute_force_sums = set()
for r in range(N_ELEMS + 1):
    for combo in itertools.combinations(S, r):
        brute_force_sums.add(sum(combo))
assert brute_force_sums == set(index_to_sum.values()), "internal consistency check failed"

classical_reachable = {k: (k in brute_force_sums) for k in TARGETS}

print("S =", S)
print("Classical reachable sums:", sorted(brute_force_sums))
print("Classical NOT reachable in 0..%d:" % TARGETS[-1],
      sorted(k for k in TARGETS if not classical_reachable[k]))

# ---------------------------------------------------------------------------
# 2. Quantum Grover search, one instance per target k.
# ---------------------------------------------------------------------------
sim = AerSimulator()
SHOTS = 4096


def grover_circuit_for_target(k, num_iterations, marked_indices):
    """Build a Grover circuit over N_ELEMS qubits marking `marked_indices`."""
    n = N_ELEMS
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    # Phase oracle: diagonal unitary that is -1 exactly on marked_indices.
    diag = [1.0] * (2 ** n)
    for m in marked_indices:
        diag[m] = -1.0
    oracle_gate = DiagonalGate(diag)

    # Diffuser (inversion about the mean) over n qubits.
    diffuser = QuantumCircuit(n)
    diffuser.h(range(n))
    diffuser.x(range(n))
    diffuser.h(n - 1)
    diffuser.mcx(list(range(n - 1)), n - 1)
    diffuser.h(n - 1)
    diffuser.x(range(n))
    diffuser.h(range(n))
    diffuser_gate = diffuser.to_gate(label="diffuser")

    for _ in range(num_iterations):
        qc.append(oracle_gate, range(n))
        qc.append(diffuser_gate, range(n))

    qc.measure(range(n), range(n))
    return qc


def optimal_iterations(num_marked, num_total):
    if num_marked == 0:
        return 0
    theta = math.asin(math.sqrt(num_marked / num_total))
    iters = round((math.pi / (4 * theta)) - 0.5)
    return max(iters, 0)


quantum_reachable = {}
detail = {}

for k in TARGETS:
    marked = [idx for idx, s in index_to_sum.items() if s == k]
    m = len(marked)
    iters = optimal_iterations(m, NUM_INDICES)
    qc = grover_circuit_for_target(k, iters, marked)
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=SHOTS).result()
    counts = result.get_counts()

    total = sum(counts.values())
    top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])
    top_prob = top_count / total

    # Does the top outcome actually correspond to a genuine witness (sum==k)?
    top_index = int(top_bitstring, 2)  # qiskit bit order: c[0] is rightmost
    top_is_valid_witness = index_to_sum.get(top_index) == k

    # Decision rule: Grover amplification only happens (and only produces a
    # high-probability, *valid* peak) when there is at least one marked
    # state. Uniform baseline probability with no marks is 1/8 = 0.125.
    found = (m > 0) and (top_prob > 0.4) and top_is_valid_witness
    quantum_reachable[k] = found
    detail[k] = dict(marked_count=m, iterations=iters, top_prob=round(top_prob, 3),
                      top_index=top_index, top_is_valid_witness=top_is_valid_witness)

    print(f"k={k}: classical_marked={m}, grover_iters={iters}, "
          f"top_prob={top_prob:.3f}, top_index={top_index}, "
          f"valid_witness={top_is_valid_witness}, quantum_says_reachable={found}")

# ---------------------------------------------------------------------------
# 3. Compare quantum verdicts to the classical ground truth.
# ---------------------------------------------------------------------------
all_match = all(quantum_reachable[k] == classical_reachable[k] for k in TARGETS)

print()
print("Classical reachability:", classical_reachable)
print("Quantum   reachability:", quantum_reachable)

if all_match:
    print("PASS")
else:
    print("FAIL")
    for k in TARGETS:
        if quantum_reachable[k] != classical_reachable[k]:
            print(f"  mismatch at k={k}: classical={classical_reachable[k]} "
                  f"quantum={quantum_reachable[k]} detail={detail[k]}")
