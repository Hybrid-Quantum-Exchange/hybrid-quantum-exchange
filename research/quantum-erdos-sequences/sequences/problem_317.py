"""
Erdos problem #317 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
"number: 317"): prize="no", status="open", tags=["number theory",
"unit fractions"], oeis=["N/A"]. There is no OEIS sequence id attached to
this problem in the source data, and no numeric formula is given in the
local metadata beyond the tag "unit fractions" -- so this script does NOT
use an OEIS sequence. This is an honest limitation: the task asked to
report accurately rather than fake a match to an OEIS term, and here there
is no OEIS id to match against at all.

Because the metadata does give a concrete, well-known mathematical theme
("unit fractions" in "number theory"), the closest genuine, small, finite,
computable property in that theme is Egyptian-fraction representability:

    Property tested: does the fraction 7/12 have a representation as a sum
    of exactly TWO unit fractions, 7/12 = 1/a + 1/b, with positive integers
    1 <= a, b <= 16?

This is a real, self-contained decision/search problem (finite search
space, checkable by direct computation), in the same "unit fractions"
family as the problem's tags, chosen because the actual formal statement
of Erdos #317 was not available in the local read-only clone (only the
YAML metadata row, which carries no OEIS id and no restated conjecture
text).

Classical ground truth (computed below in `classical_search`, from first
principles, no lookup): among a in [1,16], the search finds all a for
which (7/12 - 1/a) is exactly the reciprocal of a positive integer b in
[1,16]. This is verified in-script before any quantum code runs.

Quantum approach: Grover's algorithm (amplitude amplification). The 16
candidate values of `a` are indexed by a 4-qubit register. A classical
pre-pass (below) determines which indices are "marked" (satisfy the unit
fraction property); that classical answer is compiled into a standard
Grover phase-oracle (multi-controlled Z on the marked bitstrings), and the
Grover diffuser is applied the optimal number of times for this space
size. The circuit is run on the ideal AerSimulator, and the most frequently
sampled index is compared against the classical marked set to test that
Grover's algorithm actually finds a genuine solution index with high
probability.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
import numpy as np
from fractions import Fraction

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

def classical_search(target: Fraction, n_bits: int):
    """Return the sorted list of 1-indexed a in [1, 2**n_bits] such that
    target - 1/a is exactly 1/b for some positive integer b in the same
    range. Pure classical brute force, no external data."""
    limit = 2 ** n_bits
    marked = []
    for a in range(1, limit + 1):
        remainder = target - Fraction(1, a)
        if remainder <= 0:
            continue
        # remainder must be 1/b for integer b
        if remainder.numerator == 1:
            b = remainder.denominator
            if 1 <= b <= limit:
                marked.append(a)
    return marked


N_BITS = 4  # 16 candidate values of a: indices 1..16, encoded as 0..15
TARGET = Fraction(7, 12)

marked_a = classical_search(TARGET, N_BITS)
assert marked_a, "classical search found no solutions -- cannot build a meaningful oracle"

# Sanity-check every marked value directly (independent re-derivation).
for a in marked_a:
    rem = TARGET - Fraction(1, a)
    assert rem.numerator == 1 and 1 <= rem.denominator <= 2 ** N_BITS
    assert Fraction(1, a) + Fraction(1, rem.denominator) == TARGET

print(f"Classical ground truth: 7/12 = 1/a + 1/b has solutions for a in {marked_a}")
for a in marked_a:
    b = (TARGET - Fraction(1, a)).denominator
    print(f"  a={a}, b={b}  =>  1/{a} + 1/{b} = {Fraction(1, a) + Fraction(1, b)}")

# Marked computational-basis indices (0-indexed: index = a - 1).
marked_indices = sorted(a - 1 for a in marked_a)


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffuser over N_BITS qubits.
# ---------------------------------------------------------------------------

def mark_index(qc: QuantumCircuit, index: int, n_bits: int):
    """Apply a phase flip (-1) to the single computational basis state
    `index`, using X gates to map it to |11...1> and a multi-controlled Z."""
    bits = format(index, f"0{n_bits}b")
    # X on qubits whose bit is '0', so target index maps to all-ones.
    for i, bit in enumerate(reversed(bits)):
        if bit == "0":
            qc.x(i)
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    for i, bit in enumerate(reversed(bits)):
        if bit == "0":
            qc.x(i)


def build_oracle(n_bits: int, indices):
    qc = QuantumCircuit(n_bits, name="oracle")
    for idx in indices:
        mark_index(qc, idx, n_bits)
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


N = 2 ** N_BITS
M = len(marked_indices)
# Optimal number of Grover iterations for M marked items out of N.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / 4) / theta - 0.5))

qc = QuantumCircuit(N_BITS, N_BITS)
qc.h(range(N_BITS))

oracle = build_oracle(N_BITS, marked_indices)
diffuser = build_diffuser(N_BITS)

for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_BITS), range(N_BITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: rightmost classical bit is qubit 0; our indices were
# built assuming bit i (from the right of the bitstring) = qubit i, which
# matches Qiskit's little-endian classical register printing directly.
sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("\nTop measured outcomes (bitstring: count):")
for bits, cnt in sorted_counts[:5]:
    idx = int(bits, 2)
    print(f"  {bits} -> index {idx} (a={idx + 1}): {cnt}/{shots}")

top_bits, top_count = sorted_counts[0]
top_index = int(top_bits, 2)

# Fraction of shots landing on ANY marked index -- Grover should strongly
# concentrate probability there.
marked_shots = sum(cnt for bits, cnt in counts.items() if int(bits, 2) in marked_indices)
marked_fraction = marked_shots / shots

print(f"\nFraction of shots landing on a marked (valid) index: {marked_fraction:.3f}")
print(f"Most frequent measured index: {top_index} (a={top_index + 1})")
print(f"Classically marked indices: {marked_indices} (a values: {marked_a})")

verified = (top_index in marked_indices) and (marked_fraction > 0.5)

print("\nPASS" if verified else "\nFAIL")
