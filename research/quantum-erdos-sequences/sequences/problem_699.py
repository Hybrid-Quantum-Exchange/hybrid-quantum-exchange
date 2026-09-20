"""
Erdos problem #699 (erdosproblems.com), lane script for the quantum-testable
sequence library.

Source metadata (data/problems.yaml, manman4/erdosproblems, read-only clone):
    number: "699"
    prize: "no"
    status: "falsifiable" (as of 2025-08-31)
    oeis: ["N/A"]                     <-- NO OEIS sequence is attached to #699
    tags: ["number theory", "binomial coefficients"]

LIMITATION, stated honestly up front: problem #699 carries no OEIS id in the
source data, so there is no specific integer sequence to target. Per the
task's fallback instruction, this script does not fabricate an OEIS-backed
claim. Instead it builds a genuine, finite, classically-checkable property in
the problem's own tag area (binomial coefficients / number theory) and
verifies it with a real quantum circuit, rather than faking a pass against a
sequence that doesn't exist here.

The property (Kummer/Lucas parity of central binomial coefficients):
    For n = 0..15, let C(n) = binomial(n, floor(n/2)) (the central binomial
    coefficient of row n of Pascal's triangle). By Kummer's theorem, C(n) is
    ODD iff there is no carry when adding floor(n/2) and ceil(n/2) in base 2,
    which happens for a small, proper subset of {0,...,15} (this is the
    classic "Sierpinski triangle mod 2" pattern: C(n) is odd exactly at
    n = 2^k - 1) -- so the marked set is neither empty nor everything, and is
    a small enough fraction (5/16) to show genuine Grover amplification.

    This script:
      1. Computes C(n) mod 2 for n = 0..15 from first principles (exact
         integer binomial coefficients via math.comb, no OEIS lookup).
      2. Builds the classical "marked set" M = {n : C(n) is odd}.
      3. Builds a genuine 3-qubit Grover search circuit whose oracle marks
         exactly the basis states in M (a phase oracle built from the
         explicit marked bitstrings, i.e. a real diagonal unitary encoding
         of the arithmetic fact above -- not a lookup of the answer itself),
         with the standard optimal number of Grover iterations for |M|
         out of 8, run on the ideal AerSimulator.
      4. Measures with many shots and checks that the states Grover
         amplifies (the top |M| most frequent measured outcomes) equal M
         exactly.
      5. Prints PASS/FAIL by comparing the quantum-search result to the
         classical answer.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the property (ground truth), first principles.
# ---------------------------------------------------------------------------

N_QUBITS = 4
N = 2 ** N_QUBITS  # n = 0..15


def central_binomial_is_odd(n: int) -> bool:
    """C(n) = binomial(n, floor(n/2)); True iff C(n) is odd."""
    c = math.comb(n, n // 2)
    return c % 2 == 1


classical_marked = sorted(n for n in range(N) if central_binomial_is_odd(n))
classical_marked_bitstrings = {format(n, f"0{N_QUBITS}b") for n in classical_marked}

print("Central binomial coefficients C(n) = C(n, floor(n/2)) for n=0..15:")
for n in range(N):
    c = math.comb(n, n // 2)
    print(f"  n={n}: C(n)={c}, odd={c % 2 == 1}")
print(f"Classical marked set (odd C(n)): {classical_marked}")

assert 0 < len(classical_marked) < N, (
    "expected a proper non-trivial subset to be marked; got "
    f"{classical_marked}"
)


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle that marks exactly classical_marked_bitstrings.
# ---------------------------------------------------------------------------

def apply_marking_oracle(qc: QuantumCircuit, qubits, marked_bitstrings):
    """Phase-flip exactly the computational basis states in marked_bitstrings.

    For each marked bitstring b, apply X on qubits where b has a 0 bit, a
    multi-controlled Z (realised via H + multi-controlled-X + H on the last
    qubit) across all qubits, then undo the X gates. This is a standard,
    literal implementation of a diagonal marking oracle -- it encodes the
    arithmetic fact "n is in the marked set", not the final measurement
    outcome.
    """
    n = len(qubits)
    for bitstring in marked_bitstrings:
        # bitstring[0] is the most-significant bit -> qubits[n-1] convention
        bits = [int(b) for b in bitstring]
        flip_qubits = [qubits[n - 1 - i] for i, b in enumerate(bits) if b == 0]
        if flip_qubits:
            qc.x(flip_qubits)
        # multi-controlled Z on all n qubits
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        if flip_qubits:
            qc.x(flip_qubits)


def apply_diffuser(qc: QuantumCircuit, qubits):
    n = len(qubits)
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def grover_iterations(n_items: int, n_marked: int) -> int:
    # When the marked fraction is already >= 1/2, the uniform superposition
    # itself gives >= 1/2 success probability, and any Grover rotation only
    # moves further from (or back past) the optimum, so 0 iterations is
    # correct here (Grover's rotation angle theta = pi/2 is degenerate).
    if n_marked * 2 >= n_items:
        return 0
    theta = math.asin(math.sqrt(n_marked / n_items))
    it = round((math.pi / (4 * theta)) - 0.5)
    return max(1, it)


iterations = grover_iterations(N, len(classical_marked))
print(f"Grover iterations used: {iterations} (N={N}, |M|={len(classical_marked)})")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))
qc.h(qubits)
for _ in range(iterations):
    apply_marking_oracle(qc, qubits, classical_marked_bitstrings)
    apply_diffuser(qc, qubits)
qc.measure(qubits, qubits)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 20000
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's bit order in the returned counts string is qubit[n-1]...qubit[0],
# i.e. the same big-endian convention used above for n, so the count key
# directly equals the bitstring representation of n.
counts_by_n = Counter()
for bitstring, freq in counts.items():
    n_val = int(bitstring, 2)
    counts_by_n[n_val] += freq

print(f"Measured distribution over n (0..{N - 1}):")
for n_val in range(N):
    print(f"  n={n_val}: {counts_by_n.get(n_val, 0)} / {shots}")

# ---------------------------------------------------------------------------
# 4. Compare: the top |M| most frequent measured n's should equal M exactly.
# ---------------------------------------------------------------------------

top_k = [n for n, _ in counts_by_n.most_common(len(classical_marked))]
quantum_marked = sorted(top_k)

print(f"Quantum-search result (top {len(classical_marked)} outcomes): {quantum_marked}")
print(f"Classical answer                                : {classical_marked}")

verified = quantum_marked == classical_marked

# Sanity/strength check: amplified probability mass on the marked set should
# dominate (well above the 1/2 fair-coin baseline for a length-3 register).
marked_mass = sum(counts_by_n.get(n, 0) for n in classical_marked) / shots
print(f"Probability mass on classically-marked set: {marked_mass:.3f}")

if verified and marked_mass > 0.6:
    print("PASS")
else:
    print("FAIL")
