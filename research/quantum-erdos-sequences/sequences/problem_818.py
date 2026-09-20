"""
Erdos problem #818 -- quantum-testable instance.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: 818"):
    prize: no
    informal_status: proved
    formal_status: Lean
    oeis: ["N/A"]
    tags: ["additive combinatorics"]

LIMITATION, stated up front: problem #818 carries no OEIS sequence id ("N/A"
in the source data), so there is no literal OEIS term to reproduce and no
oracle can be built "from problem 818's own sequence". This script is the
best honest substitute: it builds a genuine, finite, computable instance of
the *kind* of object problem 818's single tag names -- additive
combinatorics, specifically 3-term-arithmetic-progression-free subsets of
{0, ..., n-1}, the same family Erdos/Behrend-type problems in this area are
about -- and verifies a real Grover search circuit against it. Nothing here
is presented as problem 818's own answer; it is a tag-matched, self-contained
quantum-testable property in the same subfield.

Classical property being tested
--------------------------------
Let n = 5, universe U = {0, 1, 2, 3, 4}. For each of the 2**n = 32 subsets
S of U (encoded as a 5-bit string, bit i set iff i in S), S is "3-AP-free"
if there is no triple i < j < k in S with j - i == k - j (no non-trivial
3-term arithmetic progression inside S).

The script:
  1. Brute-forces, in plain Python, every one of the 32 subsets, computes
     which are 3-AP-free, and finds their maximum size m* and the exact set
     M of subsets attaining that maximum size (this is the classical answer,
     derived here from first principles, not copied from anywhere).
  2. Builds a genuine Grover search circuit over 5 qubits whose oracle flips
     the phase of exactly the computational basis states in M (a diagonal
     unitary built from the classical truth table above -- a standard,
     legitimate way to realize an oracle once the marked set is known), with
     the standard Grover diffusion operator, run for the optimal number of
     Grover iterations for |M| marked items out of 32.
  3. Runs the circuit on the ideal AerSimulator, takes the most-sampled
     bitstring(s), and checks they are members of M (i.e. that Grover found
     a genuine maximum 3-AP-free subset), comparing quantum output to the
     classical answer computed in step 1.

PASS means: the most frequent measurement outcome(s) from the quantum circuit
are exactly maximum-size 3-AP-free subsets, matching the classical brute
force.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator


N = 5  # universe size {0,...,N-1}
DIM = 1 << N  # 32 subsets


def is_3ap_free(subset_bits):
    """subset_bits: sorted list of ints in {0,...,N-1}. True iff no 3-AP."""
    s = set(subset_bits)
    for i, j, k in combinations(sorted(s), 3):
        if j - i == k - j:
            return False
    return True


def bits_of(mask, n=N):
    return [b for b in range(n) if (mask >> b) & 1]


# ---- Step 1: classical brute force over all 32 subsets ----
sizes = {}
free_masks = []
for mask in range(DIM):
    elems = bits_of(mask)
    if is_3ap_free(elems):
        free_masks.append(mask)
        sizes[mask] = len(elems)

max_size = max(sizes.values())
marked = sorted(m for m, s in sizes.items() if s == max_size)

print(f"Classical brute force over n={N} ({DIM} subsets):")
print(f"  3-AP-free subsets found: {len(free_masks)}")
print(f"  maximum 3-AP-free size m* = {max_size}")
print(f"  maximizing subsets (bitmasks): {marked}")
for m in marked:
    print(f"    mask={m:05b} -> set={sorted(bits_of(m))}")

assert len(marked) >= 1
M = len(marked)

# ---- Step 2: build Grover circuit ----
# Diagonal oracle: -1 phase on each state in `marked`, +1 elsewhere.
diag = np.ones(DIM, dtype=complex)
for m in marked:
    diag[m] = -1.0

oracle = QuantumCircuit(N, name="Oracle")
oracle.append(DiagonalGate(list(diag)), list(range(N)))

# Standard diffusion operator (inversion about the mean) for N qubits.
diffusion = QuantumCircuit(N, name="Diffusion")
diffusion.h(range(N))
diffusion.x(range(N))
diffusion.h(N - 1)
diffusion.mcx(list(range(N - 1)), N - 1)
diffusion.h(N - 1)
diffusion.x(range(N))
diffusion.h(range(N))

# Optimal iteration count for M marked out of DIM items.
theta = np.arcsin(np.sqrt(M / DIM))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffusion, inplace=True)
qc.measure(range(N), range(N))

print(f"\nGrover circuit: N={N} qubits, |marked|={M}, iterations={iterations}")

# ---- Step 3: run on ideal AerSimulator ----
sim = AerSimulator()
qc = transpile(qc, sim, basis_gates=["u", "cx"])
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order: rightmost char is qubit 0.
def outcome_to_mask(bitstring):
    return int(bitstring[::-1], 2)

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_count = sorted_counts[0][1]
top_outcomes = [outcome_to_mask(b) for b, c in sorted_counts if c == top_count]
# also report a small top-k for visibility
print("Top measured outcomes:")
for b, c in sorted_counts[:min(6, len(sorted_counts))]:
    mask = outcome_to_mask(b)
    print(f"  bits={b} mask={mask:05b} set={sorted(bits_of(mask))} count={c}")

quantum_found_marked = all(m in marked for m in top_outcomes)
# amplitude amplification on the marked subspace should concentrate most
# shots there; require a strong majority land on marked states too.
marked_shot_fraction = sum(c for b, c in counts.items() if outcome_to_mask(b) in marked) / shots

print(f"\nMost-frequent outcome(s) are among classical maximum 3-AP-free sets: {quantum_found_marked}")
print(f"Fraction of shots landing on a marked (maximum 3-AP-free) subset: {marked_shot_fraction:.3f}")

verified = quantum_found_marked and marked_shot_fraction > 0.5

if verified:
    print("\nPASS: quantum Grover search result matches classical answer.")
else:
    print("\nFAIL: quantum Grover search result does not match classical answer.")

assert verified, "Quantum result did not verify against classical brute force."
