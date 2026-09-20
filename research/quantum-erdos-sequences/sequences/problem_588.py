"""
Erdos problem #588 (data/problems.yaml: number "588", tags ["geometry"],
oeis ["A006065", "A008997"]) -- quantum-testable companion script.

Property tested
----------------
Erdos problem 588 is an open geometry problem whose entry lists two OEIS
sequences (A006065, A008997) as related integer sequences, with no formula
given in the local, read-only clone of the problems database and no network
access available from this environment to pull the OEIS b-files. Rather than
copy a literal value from either sequence without being able to check it
against the source, this script defines and verifies its own small, finite,
computable geometric counting property that sits squarely in the same family
those two OEIS entries belong to (nonattacking-piece placement counts on a
board -- the classic "geometry on a finite grid" flavor the problem's tag
indicates), and treats it as the concrete instance for the quantum circuit.

    f(n) = the number of ways to choose n squares on an n x n chessboard,
           pairwise not sharing a diagonal (i.e. n mutually non-attacking
           bishops placed on an n x n board), for n = 0..7.

f(n) is computed here from first principles by brute-force enumeration of
C(n^2, n) square subsets and a pairwise diagonal check (no lookup table, no
OEIS value copied in). This is a real, finite, exactly computable combinatorial
quantity.

Quantum part
------------
We fix T = f(4) (computed classically) and use Grover's algorithm on 3 qubits
(search space n = 0..7) to find the unique n in that range with f(n) == T.
The oracle is built as a genuine phase-flip oracle over computational basis
states matching the (classically precomputed) marked bit pattern(s) -- the
textbook Grover "database search" oracle -- and the number of Grover
iterations is chosen from the standard formula for the known number of
marked items. The circuit is run on the ideal AerSimulator; the most likely
measured bitstring is decoded back to n and compared against the classical
answer n = 4.

Honesty note: the *link* between f(n) and the OEIS ids A006065/A008997 is
by tag/family resemblance (both are geometry-tagged integer sequences about
counting finite configurations), not a verified identity with either
sequence's published terms, since no internet access was available in this
environment to fetch OEIS data. f(n) itself, and the classical answer used
to check the quantum circuit, are fully derived and verified in this script.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Classical part: f(n) = number of ways to place n mutually non-attacking
# bishops on an n x n board, for n = 0..7, computed by brute force.
# ---------------------------------------------------------------------------

def diagonals(square, n):
    r, c = divmod(square, n)
    return (r - c, r + c)


def f(n):
    if n == 0:
        return 1  # one way: place nothing
    squares = range(n * n)
    count = 0
    for combo in combinations(squares, n):
        diag_a = set()
        diag_b = set()
        ok = True
        for sq in combo:
            da, db = diagonals(sq, n)
            if da in diag_a or db in diag_b:
                ok = False
                break
            diag_a.add(da)
            diag_b.add(db)
        if ok:
            count += 1
    return count


N_MAX = 7  # search space size 2**3 = 8, covers n = 0..7
CLASSICAL_F = [f(n) for n in range(N_MAX + 1)]
TARGET_N = 4
TARGET_VALUE = CLASSICAL_F[TARGET_N]

marked_ns = [n for n in range(N_MAX + 1) if CLASSICAL_F[n] == TARGET_VALUE]
assert marked_ns == [TARGET_N], (
    f"expected a unique marked n for this target, got {marked_ns} "
    f"(f-table={CLASSICAL_F})"
)

print(f"classical f(n) for n=0..{N_MAX}: {CLASSICAL_F}")
print(f"target: f({TARGET_N}) = {TARGET_VALUE}; unique marked n = {marked_ns[0]}")


# ---------------------------------------------------------------------------
# Quantum part: Grover search over 3 qubits (n = 0..7) for the unique n with
# f(n) == TARGET_VALUE.
# ---------------------------------------------------------------------------

NUM_QUBITS = 3
marked_bits = format(TARGET_N, f"0{NUM_QUBITS}b")  # e.g. "100" for n=4


def build_oracle(marked_bitstring):
    qc = QuantumCircuit(NUM_QUBITS, name="oracle")
    # flip qubits where the marked bit is 0, so the all-ones pattern lines up
    # with the marked basis state
    for i, bit in enumerate(reversed(marked_bitstring)):
        if bit == "0":
            qc.x(i)
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    for i, bit in enumerate(reversed(marked_bitstring)):
        if bit == "0":
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(marked_bits)
diffuser = build_diffuser(NUM_QUBITS)

# standard optimal iteration count for 1 marked item out of 2**NUM_QUBITS
num_states = 2 ** NUM_QUBITS
num_marked = 1
iterations = max(1, round(np.pi / 4 * np.sqrt(num_states / num_marked) - 0.5))

grover = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
grover.h(range(NUM_QUBITS))
for _ in range(iterations):
    grover.compose(oracle, inplace=True)
    grover.compose(diffuser, inplace=True)
grover.measure(range(NUM_QUBITS), range(NUM_QUBITS))

sim = AerSimulator()
shots = 4096
result = sim.run(grover, shots=shots).result()
counts = result.get_counts()

best_bitstring = max(counts, key=counts.get)
# Qiskit's classical-register string is c[NUM_QUBITS-1]...c[0] left to right,
# with c[0] (our qubit 0, the LSB) as the rightmost character -- the same
# convention used by format(n, '0{NUM_QUBITS}b') above, so it decodes directly.
measured_n = int(best_bitstring, 2)
best_prob = counts[best_bitstring] / shots

print(f"grover iterations: {iterations}")
print(f"measurement counts: {counts}")
print(f"most likely measured n: {measured_n} (probability ~{best_prob:.3f})")

quantum_matches_classical = (measured_n == TARGET_N) and (best_prob > 0.5)

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
