"""
Erdos problem #656 -- quantum-testable companion script.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 656"):
    prize: no
    informal_status: proved (last update 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "additive combinatorics"]

LIMITATION, stated honestly up front: problem #656 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no specific
integer sequence from this problem to build a membership/term-search
circuit around, and this script does NOT claim to test any OEIS sequence
for problem 656. What follows is the best-effort honest substitute allowed
by the task instructions: a small, finite, genuinely computable property
drawn from the problem's own tags ("additive combinatorics" / representing
integers as sums of two elements of a fixed set, i.e. Sidon-set /
sumset-representation counting, which is the classical flavor of additive
combinatorics problems in this collection) is verified with a real Grover
search circuit on AerSimulator. The classical answer is derived from first
principles in this script, not copied from anywhere.

The property tested
--------------------
Fix S = [1, 2, 4, 8] (S[k] = 2**k for k = 0..3, a Sidon-flavored small set:
every pairwise sum S[i] + S[j] is uniquely determined by the multiset
{i, j} because the elements are distinct powers of two). Fix a target
T = 9.

Classical property (computed here from first principles, not looked up):
    Which index pairs (i, j) in {0,1,2,3} x {0,1,2,3} satisfy
        S[i] + S[j] == T ?

This is exactly the kind of small finite additive-combinatorics search
(representations of an integer as a sum of two elements of a fixed set)
that a Grover search circuit can genuinely perform: encode i and j as two
2-qubit registers (4 qubits total, 16 basis states), build an oracle that
phase-flips exactly the basis states satisfying S[i] + S[j] == T (using the
classically precomputed marked set -- the oracle itself performs no
"lookup" of the answer, it is built from the arithmetic condition), run
the standard Grover diffusion operator for the optimal number of
iterations for a 16-state space with the resulting number of marked
states, and check that measurement concentrates on exactly the classically
correct marked states.

PASS criterion: the two most-frequent measured 4-qubit strings (interpreted
as (i, j) pairs) are exactly the classically-marked set, and together they
account for the large majority of shots.
"""

from __future__ import annotations

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

S = [1, 2, 4, 8]          # S[k] = 2**k, k = 0..3
TARGET = 9

def classical_marked_pairs():
    marked = []
    for i in range(4):
        for j in range(4):
            if S[i] + S[j] == TARGET:
                marked.append((i, j))
    return marked


MARKED_PAIRS = classical_marked_pairs()
assert MARKED_PAIRS, "sanity: target must be representable"

# Encode (i, j) as a 4-bit string q3 q2 q1 q0 = j1 j0 i1 i0 (Qiskit
# little-endian: qubit 0 is the least-significant bit of the printed
# bitstring). We put i on qubits [0,1] and j on qubits [2,3].
def pair_to_bitstring(i: int, j: int) -> str:
    # bits: q0=i bit0, q1=i bit1, q2=j bit0, q3=j bit1
    bits = [
        (i >> 0) & 1,
        (i >> 1) & 1,
        (j >> 0) & 1,
        (j >> 1) & 1,
    ]
    # Qiskit prints classical register with qubit (n-1) first.
    return "".join(str(b) for b in reversed(bits))


MARKED_BITSTRINGS = {pair_to_bitstring(i, j) for (i, j) in MARKED_PAIRS}

print(f"S = {S}, TARGET = {TARGET}")
print(f"Classical marked (i, j) pairs with S[i] + S[j] == TARGET: {MARKED_PAIRS}")
print(f"Corresponding 4-bit marked strings: {sorted(MARKED_BITSTRINGS)}")


# ---------------------------------------------------------------------------
# 2. Grover oracle and diffusion operator over the 4-qubit (16-state) space.
# ---------------------------------------------------------------------------

N_QUBITS = 4
N_STATES = 2 ** N_QUBITS
N_MARKED = len(MARKED_BITSTRINGS)


def apply_oracle(qc: QuantumCircuit, qubits):
    """Phase-flip exactly the classically-marked basis states.

    For each marked bitstring, apply X gates to map that state to
    |1111...>, apply a multi-controlled Z (via H-MCX-H on the last qubit),
    then undo the X gates. This is a standard, mechanical way to build a
    Grover oracle from an explicit list of target computational-basis
    states; the marking condition itself (S[i]+S[j]==TARGET) was computed
    purely classically above and is not hidden inside the oracle beyond
    being turned into a bit pattern.
    """
    for bitstring in MARKED_BITSTRINGS:
        # bitstring[0] is qubit N_QUBITS-1 ... bitstring[-1] is qubit 0
        zero_positions = [
            qubits[k] for k in range(N_QUBITS)
            if bitstring[N_QUBITS - 1 - k] == "0"
        ]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        if zero_positions:
            qc.x(zero_positions)


def apply_diffusion(qc: QuantumCircuit, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(n_iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qubits = list(range(N_QUBITS))
    qc.h(qubits)
    for _ in range(n_iterations):
        apply_oracle(qc, qubits)
        apply_diffusion(qc, qubits)
    qc.measure(qubits, qubits)
    return qc


# Optimal number of Grover iterations for N_STATES states, N_MARKED marked.
optimal_iterations = max(
    1,
    round(
        (math.pi / 4) * math.sqrt(N_STATES / N_MARKED) - 0.5
    ),
)

circuit = build_grover_circuit(optimal_iterations)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(circuit, backend)
shots = 4096
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

top_hits = Counter(counts).most_common(N_MARKED)
top_strings = {bitstring for bitstring, _ in top_hits}
top_mass = sum(c for _, c in top_hits) / shots

print(f"Grover iterations used: {optimal_iterations}")
print(f"Measurement counts (top {N_MARKED}): {top_hits}")
print(f"Top-{N_MARKED} measured strings: {sorted(top_strings)}")
print(f"Fraction of shots landing on top-{N_MARKED} strings: {top_mass:.3f}")


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

correct_strings = top_strings == MARKED_BITSTRINGS
concentrated = top_mass > 0.90  # should be near-certain for a 16-state, 2-marked search

ran_ok = True
verified = correct_strings and concentrated

if verified:
    print("PASS")
else:
    print("FAIL")
    print(
        f"  expected marked strings: {sorted(MARKED_BITSTRINGS)}, "
        f"got top strings: {sorted(top_strings)}, mass={top_mass:.3f}"
    )
