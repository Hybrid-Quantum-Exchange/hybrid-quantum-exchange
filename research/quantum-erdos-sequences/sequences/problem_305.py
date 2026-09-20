"""
Erdos problem #305 — quantum-testable instance.

Source metadata (erdosproblems.com data, as cloned locally):
    number: "305"
    tags: ["number theory", "unit fractions"]
    oeis: ["possible"]   <-- NOT a real OEIS sequence id. The data file's
    "oeis" field for problem 305 holds the literal string "possible", which
    is not a valid A-number and does not identify any actual OEIS entry.
    There is therefore no genuine OEIS sequence to target for this problem.

Honest limitation, stated up front: because no real OEIS id is available for
problem #305, this script does not (and cannot honestly claim to) verify a
term of "the" OEIS sequence for problem 305. Instead, since the problem's
tags are "number theory" / "unit fractions", this script builds a genuine,
self-contained finite/computable problem squarely in that area — the
classic Egyptian-fraction / unit-fraction question "does some subset of a
small finite set of unit fractions sum to exactly 1?" — and solves it with
a real Grover search circuit on the ideal AerSimulator. This is an honest
best-effort quantum-testable companion piece inspired by the problem's
subject matter, not a verification of a specific Erdos-problems.com claim
or a specific OEIS term (since none is citable here).

Classical property being tested
--------------------------------
Universe of candidate denominators: D = [2, 3, 4, 5, 6]  (5 elements).
Search space: all 2^5 = 32 subsets S of D, encoded as 5-bit strings
(bit i set <=> D[i] in S), excluding the empty set.

Property: subset S satisfies sum_{d in S} 1/d == 1 exactly (as an exact
rational, via Python's Fraction — no floating point).

This is computed here from first principles by brute force over all 32
subsets using exact rational arithmetic (fractions.Fraction), independent
of qiskit. The classically-found solution set is then encoded as marked
computational basis states of a Grover search circuit, and the circuit is
run to confirm Grover search actually finds (with high probability) a
marked state from that same solution set on the ideal AerSimulator.

Known solutions among subsets of {2,3,4,5,6} with reciprocals summing to 1:
    {2,3,6}:       1/2 + 1/3 + 1/6 = 1
    {2,4,5,... }:  checked exhaustively below (no assumptions)

Circuit: standard Grover's algorithm.
    - 5 qubits (one per candidate denominator, 2^5 = 32 basis states).
    - Oracle: phase-flips exactly the basis states corresponding to the
      classically-found solution subsets (multi-controlled Z gates, with
      X-conjugation to address the 0-bits of each marked bitstring).
    - Diffuser: standard Grover diffusion operator (inversion about the
      mean).
    - Number of iterations: floor(pi/4 * sqrt(N/M)), N=32, M=|solutions|.

Verification: after running the circuit on AerSimulator, the script checks
that the measurement outcome with the highest probability is one of the
classically-computed solution bitstrings, and that the aggregate
probability mass on solution states is large (Grover amplification
worked). PASS/FAIL is printed based on this comparison against the
independently-computed classical answer.
"""

import sys
from fractions import Fraction
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (ground truth), from first principles.
# ---------------------------------------------------------------------------

DENOMINATORS = [2, 3, 4, 5, 6]
N_QUBITS = len(DENOMINATORS)
N_STATES = 2 ** N_QUBITS


def bitstring_to_subset(bits):
    """bits: tuple of 0/1 of length N_QUBITS, bit i <-> DENOMINATORS[i]."""
    return [DENOMINATORS[i] for i, b in enumerate(bits) if b == 1]


def subset_sums_to_one(bits):
    if all(b == 0 for b in bits):
        return False
    total = Fraction(0, 1)
    for i, b in enumerate(bits):
        if b == 1:
            total += Fraction(1, DENOMINATORS[i])
    return total == Fraction(1, 1)


classical_solutions = []  # list of bitstrings (tuples), index i -> qubit i
for combo_bits in range(N_STATES):
    bits = tuple((combo_bits >> i) & 1 for i in range(N_QUBITS))
    if subset_sums_to_one(bits):
        classical_solutions.append(bits)

if not classical_solutions:
    print("No unit-fraction solutions found in this search space; "
          "cannot build a meaningful search oracle.")
    sys.exit(1)

print("Classical brute-force search over subsets of", DENOMINATORS)
print("Solutions (subset -> sum):")
for bits in classical_solutions:
    subset = bitstring_to_subset(bits)
    s = sum(Fraction(1, d) for d in subset)
    print(f"  bits={bits} subset={subset} sum={s}")

M = len(classical_solutions)
print(f"\nN = {N_STATES} candidate subsets, M = {M} exact solutions.")

# integer values of solutions, for later comparison against measured ints
solution_ints = sorted(
    sum(b << i for i, b in enumerate(bits)) for bits in classical_solutions
)
print("Solution integers (little-endian bit i = qubit i):", solution_ints)


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle marking exactly the classical solutions.
# ---------------------------------------------------------------------------

def apply_marking_for_bits(qc, bits, qubits):
    """Phase-flip the basis state 'bits' using a multi-controlled Z,
    conjugating with X gates on the qubits whose bit is 0."""
    zero_qubits = [qubits[i] for i, b in enumerate(bits) if b == 0]
    for q in zero_qubits:
        qc.x(q)
    # multi-controlled Z on all N_QUBITS qubits (phase flip |11...1>)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in zero_qubits:
        qc.x(q)


def build_oracle(n_qubits, solutions):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    qubits = list(range(n_qubits))
    for bits in solutions:
        apply_marking_for_bits(qc, bits, qubits)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qubits = list(range(n_qubits))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


oracle = build_oracle(N_QUBITS, classical_solutions)
diffuser = build_diffuser(N_QUBITS)

num_iterations = max(1, round((np.pi / 4) * np.sqrt(N_STATES / M)))
print(f"\nGrover iterations: {num_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(num_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 4096
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first as classical-register string; classical
# register bit order corresponds to measure(range(N), range(N)) i.e.
# c[i] = q[i], and the printed string has c[n-1] ... c[0]. Convert back to
# our little-endian "bit i = qubit i" integer encoding explicitly.
def counts_key_to_int(key):
    # key is a string like 'b4 b3 b2 b1 b0' order (MSB first) with no spaces
    rev = key[::-1]  # now rev[i] = qubit i's bit
    return sum(int(rev[i]) << i for i in range(len(rev)))

measured_ints = {}
for key, c in counts.items():
    val = counts_key_to_int(key)
    measured_ints[val] = measured_ints.get(val, 0) + c

sorted_measured = sorted(measured_ints.items(), key=lambda kv: -kv[1])
print("\nTop measured outcomes (integer, count):")
for val, c in sorted_measured[:8]:
    tag = " <-- classical solution" if val in solution_ints else ""
    print(f"  {val:2d} (bits {tuple((val>>i)&1 for i in range(N_QUBITS))}): "
          f"{c}/{shots}{tag}")

solution_mass = sum(c for val, c in measured_ints.items()
                     if val in solution_ints)
solution_fraction = solution_mass / shots

top_val, top_count = sorted_measured[0]
top_is_solution = top_val in solution_ints

print(f"\nProbability mass on classical solution states: "
      f"{solution_fraction:.3f} (uniform-random baseline would be "
      f"{M / N_STATES:.3f})")
print(f"Most probable measured outcome is a classical solution: "
      f"{top_is_solution}")


# ---------------------------------------------------------------------------
# 4. PASS/FAIL
# ---------------------------------------------------------------------------

verified = top_is_solution and solution_fraction > (M / N_STATES) * 2

if verified:
    print("\nPASS: Grover search on the ideal AerSimulator found a subset "
          "of {2,3,4,5,6} whose unit fractions sum to exactly 1, matching "
          "the classically brute-forced answer, with amplified probability "
          "well above the uniform-random baseline.")
else:
    print("\nFAIL: quantum search result did not match/amplify the "
          "classical answer as expected.")

sys.exit(0 if verified else 1)
