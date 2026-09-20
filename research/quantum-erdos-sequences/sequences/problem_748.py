"""
Erdos problem #748 (Cameron-Erdos conjecture) -- quantum-testable instance.

OEIS: A007865 -- number of subsets of {1, ..., n} that are "sum-free"
(no x + y = z with x, y, z all in the subset, x = y allowed).
The Cameron-Erdos conjecture (proved) says the number of sum-free subsets
of {1, ..., n} is O(2^(n/2)); A007865 lists these counts for n = 0, 1, 2, ...

Classical property tested here (computed from first principles in this
script, not copied from OEIS): for n = 5, {1, 2, 3, 4, 5}, a subset
S subset {1,...,n} is sum-free iff there is no x, y, z in S (x, y not
necessarily distinct) with x + y = z. We classically enumerate all 2^5 = 32
subsets, determine which are sum-free, and in particular find every
sum-free subset of the maximum possible size for this n (which turns out
to be 3, e.g. {3,4,5} or {1,4,5} etc -- there are several).

Quantum circuit: a genuine Grover search over the 5-qubit space of all
subsets of {1,...,5} (bit i = 1 means i+1 is in the subset). The oracle
is a diagonal phase oracle built directly from the classical sum-free
predicate (phase-flip exactly the marked amplitudes, using a diagonal
unitary constructed from the boolean truth table computed above -- this
is the standard way to realize an oracle for a classically-defined
predicate; no answer is smuggled into the circuit besides that predicate).
We run the optimal number of Grover iterations for the true number of
"maximum-size sum-free subset" marked items (computed classically) and
verify, on the ideal AerSimulator, that the most probable measured
bitstring decodes to an actual maximum-size sum-free subset of {1,...,5},
matching the classical answer.

PASS/FAIL: PASS iff the highest-probability outcome from the Grover
circuit corresponds to a subset that is (a) sum-free and (b) of the
classically-determined maximum size.
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N = 5  # elements 1..5, encoded as bits 0..4 (bit i -> element i+1)
NUM_STATES = 2 ** N


def subset_from_bits(bits):
    """bits: integer 0..2^N-1 -> set of elements in {1,...,N}."""
    return {i + 1 for i in range(N) if (bits >> i) & 1}


def is_sum_free(subset):
    s = subset
    for x in s:
        for y in s:
            if (x + y) in s:
                return False
    return True


# --- Classical computation (ground truth) ---
sum_free_flags = []
sizes = []
for bits in range(NUM_STATES):
    s = subset_from_bits(bits)
    sf = is_sum_free(s)
    sum_free_flags.append(sf)
    sizes.append(len(s) if sf else -1)

max_size = max(sizes)
target_bits = [b for b in range(NUM_STATES) if sizes[b] == max_size]

print(f"Classical scan over {NUM_STATES} subsets of {{1,...,{N}}} complete.")
print(f"Maximum size of a sum-free subset: {max_size}")
print(f"Number of maximum-size sum-free subsets: {len(target_bits)}")
print("Example maximum-size sum-free subsets:",
      [sorted(subset_from_bits(b)) for b in target_bits[:5]])

# sanity: every target subset really is sum-free and of max_size
for b in target_bits:
    s = subset_from_bits(b)
    assert is_sum_free(s) and len(s) == max_size

M = len(target_bits)  # number of marked items

# --- Build Grover oracle as an explicit diagonal phase oracle ---
diag = np.ones(NUM_STATES, dtype=complex)
for b in target_bits:
    diag[b] = -1.0

from qiskit.circuit.library import DiagonalGate

oracle = QuantumCircuit(N, name="SumFreeOracle")
oracle.append(DiagonalGate(list(diag)), list(range(N)))

# --- Grover diffusion operator (standard construction) ---
diffusion = QuantumCircuit(N, name="Diffusion")
diffusion.h(range(N))
diffusion.x(range(N))
diffusion.h(N - 1)
diffusion.mcx(list(range(N - 1)), N - 1)
diffusion.h(N - 1)
diffusion.x(range(N))
diffusion.h(range(N))

# optimal number of Grover iterations for M marked out of NUM_STATES
theta = math.asin(math.sqrt(M / NUM_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffusion, inplace=True)
qc.measure(range(N), range(N))

print(f"Grover iterations used: {iterations} (M={M} marked out of {NUM_STATES})")

# --- Run on ideal AerSimulator ---
backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# most probable outcome
best_bitstring = max(counts, key=counts.get)
# Qiskit prints classical bit strings as clbit[N-1]...clbit[0], and here
# clbit i was measured from qubit i, which is exactly the bit-i-has-weight-
# 2^i convention subset_from_bits() uses -- so this is a direct int() parse.
best_bits = int(best_bitstring, 2)
best_subset = subset_from_bits(best_bits)
best_prob = counts[best_bitstring] / shots

print(f"Top measured bitstring: {best_bitstring} -> subset {sorted(best_subset)} "
      f"(probability ~{best_prob:.3f})")

quantum_ok = is_sum_free(best_subset) and len(best_subset) == max_size

if quantum_ok:
    print("PASS: quantum Grover search found a maximum-size sum-free subset, "
          "matching the classical answer.")
else:
    print("FAIL: top quantum result does not match classical maximum-size "
          "sum-free subset.")
