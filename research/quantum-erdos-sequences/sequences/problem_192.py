"""
Erdos problem #192 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: '192'"):
    prize: no
    informal_status: solved
    formal_status: Lean
    oeis: ["N/A"]          <-- no OEIS sequence id is recorded for this problem
    tags: ["arithmetic progressions", "combinatorics"]

LIMITATION (reported honestly, per instructions): problem #192 carries no OEIS
id in the source data ("N/A"), so there is no OEIS sequence to derive a
membership/counting property from for this lane. There is therefore no
literal OEIS value to check a quantum circuit against for this problem.

BEST-EFFORT SUBSTITUTE: the problem's own tags ("arithmetic progressions",
"combinatorics") point to a small, finite, genuinely computable property in
the same mathematical neighborhood as Erdos's arithmetic-progression work
(van der Waerden-type colorings): does there exist a 2-coloring of
{0, 1, ..., n-1} (n = 8 here) with no monochromatic 3-term arithmetic
progression (3-AP)?  This is exactly decidable by brute force for small n
(it is known classically that such colorings exist for all n < W(3,2) = 9,
so n = 8 is the largest instance where a valid coloring still exists), and
it is a real combinatorial search problem, which is what Grover's algorithm
is built to accelerate.

The classical answer (search space size N=2, marked/valid colorings, and one
example) is computed here from first principles by brute-force enumeration
over all 2^n colorings -- not copied from any table.

The quantum part builds a genuine Grover search circuit whose oracle is
derived, in code, from the exact set of valid (3-AP-free) colorings found by
the classical brute-force search above (a phase oracle implemented as an
explicit diagonal unitary with -1 on marked computational-basis states, +1
elsewhere -- a standard, faithful way to realize an arbitrary boolean oracle
for a Grover search on a simulator). It runs on the ideal AerSimulator,
applies the optimal number of Grover iterations for this search-space
size/marked-count, measures, and checks that the measured n-bit string is
indeed one of the classically-valid 3-AP-free colorings (i.e. that Grover
amplified the correct set of marked states). PASS/FAIL is decided by that
check, run many times to get a high-confidence success probability.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical part: brute-force, from first principles.
# ---------------------------------------------------------------------------

N = 8  # coloring of {0, ..., N-1}; W(3,2) = 9, so N=8 is the largest instance
        # with a guaranteed-existing valid coloring, keeping the search space
        # (2^N = 256) small enough for a few-qubit Grover circuit.


def has_mono_3ap(coloring):
    """True if `coloring` (tuple of 0/1, length N) contains a monochromatic
    3-term arithmetic progression i, i+d, i+2d (d >= 1) with all three
    positions the same color."""
    n = len(coloring)
    for d in range(1, (n - 1) // 2 + 1):
        for i in range(0, n - 2 * d):
            a, b, c = coloring[i], coloring[i + d], coloring[i + 2 * d]
            if a == b == c:
                return True
    return False


def brute_force_valid_colorings(n):
    valid = []
    for bits in itertools.product((0, 1), repeat=n):
        if not has_mono_3ap(bits):
            valid.append(bits)
    return valid


valid_colorings = brute_force_valid_colorings(N)
num_marked = len(valid_colorings)
search_space_size = 2 ** N

assert num_marked > 0, "classical claim (a valid coloring exists for N=8) failed"

# Represent each valid coloring as its integer index (bit i = coloring[i],
# little-endian to match Qiskit's qubit ordering) for use as the oracle's
# marked-state set.
marked_indices = sorted(
    int("".join(str(b) for b in reversed(bits)), 2) for bits in valid_colorings
)

print(f"Classical brute force over N={N}, search space size = {search_space_size}")
print(f"Number of 3-AP-free 2-colorings found: {num_marked}")
print(f"Example valid coloring: {valid_colorings[0]}")


# ---------------------------------------------------------------------------
# 2. Quantum part: Grover search over the N-bit space for a 3-AP-free coloring.
# ---------------------------------------------------------------------------

n_qubits = N  # 8 qubits, one per position in the coloring


def build_oracle(marked, dim):
    """Diagonal phase oracle: -1 on marked computational-basis indices,
    +1 elsewhere. This is an exact, faithful realization of the boolean
    oracle f(x) = 1 iff x is a 3-AP-free coloring."""
    diag = np.ones(dim, dtype=complex)
    for idx in marked:
        diag[idx] = -1.0
    return Operator(np.diag(diag))


def build_diffuser(dim):
    """Standard Grover diffusion operator: 2|s><s| - I, where |s> is the
    uniform superposition."""
    s = np.ones((dim, 1), dtype=complex) / math.sqrt(dim)
    reflect = 2 * (s @ s.conj().T) - np.eye(dim, dtype=complex)
    return Operator(reflect)


oracle_op = build_oracle(marked_indices, search_space_size)
diffuser_op = build_diffuser(search_space_size)

# Optimal number of Grover iterations for M marked items out of N_total.
theta = math.asin(math.sqrt(num_marked / search_space_size))
optimal_iters = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(optimal_iters):
    qc.unitary(oracle_op, range(n_qubits), label="oracle")
    qc.unitary(diffuser_op, range(n_qubits), label="diffuser")
qc.measure(range(n_qubits), range(n_qubits))

print(f"Grover iterations used: {optimal_iters}")

backend = AerSimulator()
shots = 2000
result = backend.run(qc, shots=shots).result()
counts = result.get_counts()

marked_set = set(marked_indices)


def bitstring_to_index(bs):
    # Qiskit's classical-register bit order is c[n-1] ... c[1] c[0] in the
    # returned key string, matching the little-endian qubit->index mapping
    # used to build marked_indices above.
    return int(bs, 2)


success_shots = sum(
    cnt for bs, cnt in counts.items() if bitstring_to_index(bs) in marked_set
)
success_prob = success_shots / shots

# Baseline: uniform random guessing would succeed with probability
# num_marked / search_space_size.
baseline_prob = num_marked / search_space_size

print(f"Measured success probability (marked outcome): {success_prob:.4f}")
print(f"Uniform-random baseline probability: {baseline_prob:.4f}")

# Also verify the single most-likely measured outcome directly encodes a
# valid (3-AP-free) coloring, checked against the classical predicate again.
most_likely_bs = max(counts, key=counts.get)
most_likely_idx = bitstring_to_index(most_likely_bs)
most_likely_bits = tuple(int(c) for c in reversed(most_likely_bs))
classical_check = not has_mono_3ap(most_likely_bits)

verified = (
    success_prob > 3 * baseline_prob  # Grover amplified the marked subspace
    and most_likely_idx in marked_set  # top outcome is genuinely marked
    and classical_check  # re-verified against the classical predicate directly
)

print(f"Most likely measured coloring: {most_likely_bits} -> 3-AP-free: {classical_check}")

if verified:
    print("PASS")
else:
    print("FAIL")
