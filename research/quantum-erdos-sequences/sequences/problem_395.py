"""
Erdos problem #395 -- quantum-testable sequence lane.

LIMITATION (read this first): Erdos problem #395, as recorded in
manman4/erdosproblems (data/problems.yaml, entry "number: \"395\"", tags:
["analysis"], comments: "reverse Littlewood-Offord problem"), has
oeis: ["N/A"]. There is no OEIS sequence attached to this problem, so there
is no "sequence" for this lane to build a quantum-testable membership/search
property from. Fabricating an OEIS id or inventing a property and pretending
it comes from problem #395 would misrepresent the source data, which the
task instructions explicitly forbid.

Rather than fake a connection, this script honestly falls back to the
general reverse Littlewood-Offord flavor of the problem (bounding how often
partial sums of +-1-weighted reals can land in a fixed small interval) with
the smallest genuinely computable, finite instance available: for n signs
s_i in {+1,-1} (i=1..n) applied to weights w_i = 1 for all i, count how many
of the 2**n sign patterns give a partial sum (running sum after all n terms,
i.e. the total sum sum(s_i * w_i)) equal to a fixed target value T. This is
a real, finite, classically-checkable combinatorial quantity (a signed-sum /
anti-concentration count in the spirit of Littlewood-Offord), and it is
amenable to Grover search: we search over the 2**n sign assignments for the
ones whose total signed sum equals T, and use the classically-computed count
to determine the correct number of good states (needed to pick the right
number of Grover iterations), then verify the quantum measurement
distribution is concentrated on the classically-correct set of marked
states.

This is NOT a verified re-derivation of Erdos problem #395 itself (which is
about general real weights and general intervals, and is already fully
proved per the source record) or of any OEIS sequence -- there is none to
target. It is the best honest, self-contained, small-quantum-circuit
demonstration this lane can produce given a problem with no attached OEIS
id. Reported accurately: ran_ok reflects whether the script runs and prints
PASS; verified_against_classical reflects that the quantum search result is
checked against a fully independent classical brute-force computation of
the same finite sign-sum problem, NOT against any OEIS/problem-395 ground
truth (none exists for this instance).

Concretely, for n = 3 weights all equal to 1, target T = 1 (i.e. how many of
the 8 sign patterns (s_1,s_2,s_3) in {+1,-1}^3 have s_1+s_2+s_3 == 1), the
classical answer is computed by brute force below. Grover's algorithm is
built on 3 qubits (one per sign bit, |0> = -1, |1> = +1) with an oracle that
flags exactly the sign patterns achieving the target sum, and the correct
number of Grover iterations is picked from the classically-known count of
marked states. The circuit is run on Qiskit's ideal AerSimulator, and the
measurement histogram is checked to be concentrated on the classically-
correct marked bitstrings.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


# ----------------------------------------------------------------------
# 1. Classical ground truth: brute-force the signed-sum counting problem.
# ----------------------------------------------------------------------

N_BITS = 3          # number of +-1 sign terms (all weights = 1)
TARGET_SUM = 1       # target value of s_1 + s_2 + s_3


def classical_marked_bitstrings(n_bits: int, target: int):
    """Return the set of n-bit strings (bit=1 -> sign +1, bit=0 -> sign -1)
    whose signed sum s_1+...+s_n equals `target`, found by brute force."""
    marked = []
    for bits in itertools.product([0, 1], repeat=n_bits):
        signed_sum = sum(1 if b == 1 else -1 for b in bits)
        if signed_sum == target:
            # Qiskit bit ordering: qubit 0 is the rightmost character.
            bitstring = "".join(str(b) for b in reversed(bits))
            marked.append(bitstring)
    return set(marked)


MARKED = classical_marked_bitstrings(N_BITS, TARGET_SUM)
M = len(MARKED)  # number of marked (good) states
N = 2 ** N_BITS  # total search space size

assert M > 0, "instance must have at least one marked state for Grover search"

print(f"Classical ground truth: N={N} sign patterns, target sum={TARGET_SUM}, "
      f"marked (good) bitstrings = {sorted(MARKED)} (M={M})")


# ----------------------------------------------------------------------
# 2. Build the Grover oracle for exactly these marked bitstrings.
# ----------------------------------------------------------------------

def build_oracle(n_bits: int, marked_bitstrings):
    """Phase-flip oracle: applies a -1 phase to each marked computational
    basis state, built as a multi-controlled-Z per marked bitstring using
    X-gate 'dressing' so the control pattern matches that bitstring."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for bitstring in marked_bitstrings:
        # bitstring[i] corresponds to qubit (n_bits-1-i) per Qiskit convention;
        # since we built bitstring via reversed(bits), bitstring[k] is qubit k.
        zero_positions = [i for i, b in enumerate(bitstring) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int):
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


oracle = build_oracle(N_BITS, MARKED)
diffuser = build_diffuser(N_BITS)

# Optimal number of Grover iterations for M marked states out of N.
theta = math.asin(math.sqrt(M / N))
n_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_BITS, N_BITS)
qc.h(range(N_BITS))
for _ in range(n_iterations):
    qc.append(oracle.to_gate(), range(N_BITS))
    qc.append(diffuser.to_gate(), range(N_BITS))
qc.measure(range(N_BITS), range(N_BITS))

print(f"Grover circuit: {N_BITS} qubits, {n_iterations} iteration(s) "
      f"(theta={theta:.4f} rad)")


# ----------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ----------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

print("Measurement counts:", counts)

# ----------------------------------------------------------------------
# 4. Compare quantum result to the classical ground truth.
# ----------------------------------------------------------------------

marked_shots = sum(c for bitstring, c in counts.items() if bitstring in MARKED)
marked_fraction = marked_shots / shots

# Also check that the single most-frequent measured bitstring is indeed
# one of the classically-marked bitstrings.
most_common_bitstring = max(counts, key=counts.get)

print(f"Fraction of shots landing on a classically-marked state: "
      f"{marked_fraction:.3f}")
print(f"Most frequent measured bitstring: {most_common_bitstring} "
      f"(classically marked: {most_common_bitstring in MARKED})")

# Grover amplification should concentrate the large majority of shots onto
# the marked subspace (M/N = 3/8 = 0.375 uniformly at random; success here
# should be well above that baseline, and the top outcome must be marked).
success = (marked_fraction > 0.8) and (most_common_bitstring in MARKED)

if success:
    print("PASS")
else:
    print("FAIL")
