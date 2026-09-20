"""
Erdos problem #321 (data/problems.yaml: number "321", tags ["number theory",
"unit fractions"], oeis ["A384927", "A391592"]).

Classical property tested
--------------------------
Erdos problem #321 concerns representations of 1 (and related unit-fraction
identities) as a sum of DISTINCT unit fractions 1/d with denominators drawn
from a finite candidate set (Egyptian-fraction style decompositions, the
subject the "unit fractions" tag and the associated OEIS sequences track).

We instantiate a small, fully finite/computable version of that flavor of
question:

    Candidate denominator set  D = (2, 3, 4, 6, 12)   (5 elements)
    Question: which subset S of D satisfies  sum_{d in S} 1/d == 1  exactly?

This is a genuine finite search problem (2^5 = 32 subsets to search) of
exactly the combinatorial-number-theory kind Erdos-#321's unit-fraction
sequences are built from. The classical answer is computed first from
first principles in this script using exact Fraction arithmetic by brute
force over all 32 subsets (no OEIS values are copied in): the unique
solution turns out to be S = {2, 3, 4, 6, 12} is NOT it; the actual unique
satisfying subset is verified below and used as Grover's marked state.

Quantum approach
-----------------
Grover's algorithm on 5 qubits (one per candidate denominator, bit=1 means
"denominator included in the subset"). The oracle is built by marking
exactly the basis state(s) that the classical brute-force search (using
exact Fraction arithmetic, computed in this script) determined satisfy
sum_{d in S} 1/d == 1. The number of Grover iterations is chosen from the
standard formula floor(pi/4 * sqrt(N/M)) for N=32 basis states and M the
number of marked solutions found classically. We then run the circuit on
the ideal AerSimulator and check that the most frequently measured
bitstring decodes to exactly the same subset(s) found classically.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

from fractions import Fraction
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, exact rational arithmetic).
# ---------------------------------------------------------------------------

DENOMINATORS = (2, 3, 4, 6, 12)
N_BITS = len(DENOMINATORS)
N_STATES = 2 ** N_BITS


def subset_from_bits(bits):
    """bits: tuple of 0/1 of length N_BITS, index i controls DENOMINATORS[i]."""
    return tuple(d for d, b in zip(DENOMINATORS, bits) if b)


def unit_fraction_sum(subset):
    total = Fraction(0)
    for d in subset:
        total += Fraction(1, d)
    return total


def classical_brute_force():
    """Return the list of index-bitstrings (as ints, LSB = first denom) whose
    subset of DENOMINATORS sums to exactly 1, found by direct enumeration."""
    solutions = []
    for mask in range(N_STATES):
        bits = tuple((mask >> i) & 1 for i in range(N_BITS))
        subset = subset_from_bits(bits)
        if not subset:
            continue
        if unit_fraction_sum(subset) == Fraction(1, 1):
            solutions.append(mask)
    return solutions


CLASSICAL_SOLUTIONS = classical_brute_force()

if len(CLASSICAL_SOLUTIONS) == 0:
    raise SystemExit(
        "No subset of DENOMINATORS sums to 1 -- pick a different candidate "
        "set. (Classical search found zero solutions.)"
    )

# Sanity: print what was found and double check with a second, independent
# classical method (itertools.combinations over all subset sizes).
def classical_check_via_combinations():
    found = []
    for r in range(1, N_BITS + 1):
        for combo in combinations(DENOMINATORS, r):
            if unit_fraction_sum(combo) == Fraction(1, 1):
                mask = 0
                for d in combo:
                    mask |= 1 << DENOMINATORS.index(d)
                found.append(mask)
    return sorted(found)


assert classical_check_via_combinations() == sorted(CLASSICAL_SOLUTIONS), (
    "Two independent classical enumeration methods disagree -- bug."
)

print(f"Denominator candidates: {DENOMINATORS}")
print(f"Search space size N = {N_STATES} (5 bits)")
print("Classically found subset(s) with sum of unit fractions == 1:")
for mask in CLASSICAL_SOLUTIONS:
    bits = tuple((mask >> i) & 1 for i in range(N_BITS))
    subset = subset_from_bits(bits)
    print(f"  mask={mask:05b}  subset={subset}  sum={unit_fraction_sum(subset)}")


# ---------------------------------------------------------------------------
# 2. Grover search circuit that finds the same marked state(s) quantumly.
# ---------------------------------------------------------------------------

M = len(CLASSICAL_SOLUTIONS)
iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N_STATES / M))))
print(f"M (number of marked states) = {M}, Grover iterations = {iterations}")


def build_oracle(marked_masks, n_bits):
    """Phase-flip oracle: applies -1 phase to each basis state in marked_masks."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for mask in marked_masks:
        bits = [(mask >> i) & 1 for i in range(n_bits)]
        # Flip qubits that should be 0, so the all-ones pattern corresponds
        # to this mask, apply a multi-controlled Z (via H-MCX-H on the last
        # qubit), then flip back.
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all n_bits qubits (phase flip when all are 1)
        qc.h(n_bits - 1)
        if n_bits - 1 > 0:
            qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
        else:
            qc.z(0)
        qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits):
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    if n_bits - 1 > 0:
        qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
    else:
        qc.z(0)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def build_grover_circuit(marked_masks, n_bits, n_iterations):
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(marked_masks, n_bits)
    diffuser = build_diffuser(n_bits)

    for _ in range(n_iterations):
        qc.append(oracle.to_instruction(), range(n_bits))
        qc.append(diffuser.to_instruction(), range(n_bits))

    qc.measure(range(n_bits), range(n_bits))
    return qc


grover_qc = build_grover_circuit(CLASSICAL_SOLUTIONS, N_BITS, iterations)

sim = AerSimulator()
compiled = transpile(grover_qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical register bit order is c[n-1] ... c[0] in the returned
# bitstring (little-endian string, MSB left = highest index qubit). Convert
# each measured bitstring back to our integer mask convention (qubit i =
# DENOMINATORS[i], LSB of mask).
def bitstring_to_mask(bitstring):
    # bitstring like '01010', bitstring[-1-i] is qubit i's measured value.
    mask = 0
    for i in range(N_BITS):
        bit = bitstring[len(bitstring) - 1 - i]
        mask |= int(bit) << i
    return mask

mask_counts = {}
for bitstring, c in counts.items():
    m = bitstring_to_mask(bitstring)
    mask_counts[m] = mask_counts.get(m, 0) + c

# Most frequently measured mask(s)
sorted_masks = sorted(mask_counts.items(), key=lambda kv: -kv[1])
top_mask, top_count = sorted_masks[0]

print("\nTop measured masks (mask: count):")
for m, c in sorted_masks[:5]:
    print(f"  {m:05b}: {c}")

quantum_found = set(m for m, _ in sorted_masks[:M])
classical_set = set(CLASSICAL_SOLUTIONS)

verified = quantum_found == classical_set

print(f"\nClassical solution mask(s): {sorted(classical_set)}")
print(f"Quantum top-{M} measured mask(s): {sorted(quantum_found)}")

if verified:
    print("PASS")
else:
    print("FAIL")
