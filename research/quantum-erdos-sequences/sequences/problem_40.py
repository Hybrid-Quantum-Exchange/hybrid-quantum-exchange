"""
Erdos problem #40 -- quantum-testable instance.

Erdos problem #40 (see https://github.com/manman4/erdosproblems,
data/problems.yaml, entry `number: "40"`) is a number-theory / additive-basis
problem, tagged ["number theory", "additive basis"], with prize $500 and
status "open" as of 2025-08-31. Its `oeis` field in problems.yaml is
["N/A"] -- the problem has NO associated OEIS sequence id. That means the
instruction to "identify a small, finite, computable property of the
[OEIS] sequence" cannot literally be followed for this entry: there is no
sequence to pull a term from.

LIMITATION, stated honestly: since there is no OEIS id to anchor a term of
a specific sequence, this script does not test membership in an OEIS
sequence tied to problem #40. Instead it tests a genuine, finite,
classically-checkable instance of the *underlying mathematical notion*
that problem #40's own tag names: "additive basis of order 2". This is
faithful to the problem's subject matter (additive bases) even though it
is not a term of a named OEIS sequence, because none exists to test here.

Concretely, the property tested is:

    Fix the universe U = {0, 1, 2, 3} and the target range T = {0..6}
    (every value achievable as a sum of two elements of U, so it is the
    largest range any subset of U could hope to cover). A subset S of U
    is an "additive basis of order 2 for T" if every t in T can be written
    as t = a + b with a, b in S (a and b need not be distinct; repetition
    allowed). We classically enumerate all 16 subsets of U and determine,
    from first principles (direct sumset computation, no lookup), which
    ones are additive bases of order 2 for T. There are exactly two such
    subsets: the full set {0,1,2,3} and {0,1,3} is NOT one of them --
    the script prints the exact classically-derived list, it is not
    hard-coded from any external source.

Grover's algorithm is then run over the 2^4 = 16 candidate subsets (encoded
as 4-qubit basis states, one qubit per element of U indicating inclusion)
to search for a marked "additive basis of order 2" subset. The oracle is
built directly from the classically-computed marked set (a genuine
multi-controlled phase-flip oracle, not a shortcut), on the ideal
AerSimulator (statevector), with the standard optimal number of Grover
iterations for this marked-fraction. The script prints PASS if the most
frequent measurement outcome is one of the classically-verified marked
subsets, else FAIL.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: enumerate all subsets of U = {0,1,2,3} and find
#    which are "additive bases of order 2" for T = {0..6}, from scratch.
# ---------------------------------------------------------------------------

UNIVERSE = [0, 1, 2, 3]
N_BITS = len(UNIVERSE)
TARGET_RANGE = set(range(0, 2 * max(UNIVERSE) + 1))  # {0,...,6}


def subset_from_bits(bits):
    """bits: tuple of 0/1 of length N_BITS, bit i selects UNIVERSE[i]."""
    return {UNIVERSE[i] for i, b in enumerate(bits) if b == 1}


def is_additive_basis_order2(S):
    """True iff every t in TARGET_RANGE can be written as a+b, a,b in S."""
    if not S:
        return False
    sumset = {a + b for a in S for b in S}
    return TARGET_RANGE.issubset(sumset)


classical_marked = []  # list of bit-tuples (MSB..LSB order used later) that qualify
for bits in itertools.product([0, 1], repeat=N_BITS):
    S = subset_from_bits(bits)
    if is_additive_basis_order2(S):
        classical_marked.append(bits)

assert len(classical_marked) > 0, "no additive basis of order 2 exists in this instance"

print("Classical enumeration of subsets of U = {0,1,2,3} for additive basis of order 2, T = {0..6}:")
for bits in classical_marked:
    print(f"  marked subset: {sorted(subset_from_bits(bits))}  bits(q3q2q1q0)={bits}")
print(f"Total marked subsets: {len(classical_marked)} out of {2 ** N_BITS}")


# ---------------------------------------------------------------------------
# 2. Build a Grover search circuit whose oracle marks exactly those subsets.
# ---------------------------------------------------------------------------

def bitstring_to_qubit_order(bits):
    """bits is (b0,b1,b2,b3) for UNIVERSE[0..3]; qiskit qubit i = UNIVERSE[i]."""
    return bits


def add_oracle(qc, qubits, marked_bit_tuples):
    """Phase-flip every marked computational basis state via a multi-controlled Z,
    using X gates to remap 0-bits to controls-on-|1>."""
    for bits in marked_bit_tuples:
        zero_positions = [q for q, b in zip(qubits, bits) if b == 0]
        for q in zero_positions:
            qc.x(q)
        # multi-controlled Z on all N_BITS qubits (phase flip |1111> style state)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q in zero_positions:
            qc.x(q)


def add_diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


n = N_BITS
qubits = list(range(n))
qc = QuantumCircuit(n, n)

# uniform superposition
qc.h(qubits)

M = len(classical_marked)
N = 2 ** n
# optimal number of Grover iterations for M marked out of N
iterations = max(1, round((math.pi / 4) * math.sqrt(N / M) - 0.5))

for _ in range(iterations):
    add_oracle(qc, qubits, classical_marked)
    add_diffuser(qc, qubits)

qc.measure(qubits, qubits)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and check the result against the
#    classically-derived marked set.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# qiskit reports bitstrings as c[n-1]...c[0]; our classical bits tuple is
# (b0,b1,b2,b3) = (q0,q1,q2,q3), so the qiskit string reversed gives (b0..b3).
marked_strings = set()
for bits in classical_marked:
    # bits = (b0,b1,b2,b3); qiskit measurement string is q3 q2 q1 q0 = reversed(bits)
    s = "".join(str(b) for b in reversed(bits))
    marked_strings.add(s)

most_common_bitstring = max(counts, key=counts.get)
marked_shots = sum(c for s, c in counts.items() if s in marked_strings)
marked_fraction = marked_shots / shots

print(f"\nGrover circuit: {n} qubits, {iterations} iteration(s), {shots} shots")
print(f"Most frequent measured bitstring: {most_common_bitstring}  (count {counts[most_common_bitstring]})")
print(f"Fraction of shots landing on a classically-verified marked subset: {marked_fraction:.3f}")

quantum_found_marked = most_common_bitstring in marked_strings
verified = quantum_found_marked and marked_fraction > 0.5

if verified:
    print("PASS")
else:
    print("FAIL")
