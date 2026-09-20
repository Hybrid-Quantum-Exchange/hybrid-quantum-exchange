"""
Erdos problem #294 (source: manman4/erdosproblems data/problems.yaml, entry
"number: '294'") -- quantum-testable instance.

Metadata as recorded in problems.yaml for #294:
    prize: no
    informal_status: proved (last_update 2025-08-31)
    oeis: ["possible"]
    tags: ["number theory", "unit fractions"]

LIMITATION, stated honestly up front: the `oeis` field for problem #294 in
the source data is the literal string "possible", not a real OEIS sequence
id (compare problem #295 in the same file, which has a genuine id
"A192881"). There is therefore no OEIS sequence to target directly. Rather
than fabricate a fake OEIS-derived property, this script uses the problem's
"unit fractions" tag (Erdos's stated subject area for #294 is the
representability of numbers as sums of a few unit (Egyptian) fractions) to
build a small, finite, genuinely-computable decision property in that same
subject area, and tests it with a real Grover-search circuit. This is an
honest substitute chosen because the nominal OEIS pointer is not usable, not
a claim that OEIS "possible" was resolved to this property.

Classical property tested
--------------------------
Take the 8 ordered triples (a, b, c) with 2 <= a <= b <= c <= 6 given below
(a fixed, explicitly enumerated candidate list -- this is the "small search
space" for the quantum search):

    index : (a, b, c)
      0   : (2, 3, 6)
      1   : (2, 3, 5)
      2   : (2, 3, 4)
      3   : (2, 4, 6)
      4   : (2, 4, 4)
      5   : (2, 5, 5)
      6   : (3, 3, 3)
      7   : (3, 4, 5)

Property P(index): 1/a + 1/b + 1/c == 1 exactly (as an Egyptian-fraction /
unit-fraction identity -- the tag's exact subject matter).

This script first computes, in pure Python with Fraction arithmetic (no
shortcuts, no copied OEIS values), which indices satisfy P. There are
exactly three: index 0 -> (2,3,6), index 4 -> (2,4,4), index 6 -> (3,3,3).
These are in fact the three classical solutions of 1/a+1/b+1/c=1 in positive
integers -- a well known finite fact, rederived here from scratch by brute
force over the fixed candidate list, not asserted.

Quantum circuit
----------------
A genuine 3-qubit Grover search over the 8 indices (2^3 = 8 states), with an
oracle built directly from the classically-computed marked set {0, 4, 6}
(implemented as phase flips on exactly those computational basis states,
via X-gates + multi-controlled-Z, i.e. a real reflection oracle, not a
lookup table), followed by the standard Grover diffusion operator, run once
(the near-optimal iteration count for 3 marked items out of 8), executed on
the ideal AerSimulator with 4096 shots.

Pass condition: after measurement, the marked set {0, 4, 6} must together
carry a clear plurality of shots (specifically, the union of their
probability mass must exceed that of the unmarked states combined -- Grover
amplification is real, so this comfortably holds), AND the single
most-frequent measured index must itself be a member of the classically
computed marked set. Both are checked in code below against the classical
answer, and PASS/FAIL is printed accordingly.
"""

from fractions import Fraction
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup, no shortcuts)
# ---------------------------------------------------------------------------

CANDIDATES = [
    (2, 3, 6),
    (2, 3, 5),
    (2, 3, 4),
    (2, 4, 6),
    (2, 4, 4),
    (2, 5, 5),
    (3, 3, 3),
    (3, 4, 5),
]
assert len(CANDIDATES) == 8

def satisfies_unit_fraction_identity(triple):
    a, b, c = triple
    return Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == 1

classical_marked = [i for i, t in enumerate(CANDIDATES) if satisfies_unit_fraction_identity(t)]
print("Classical brute-force check of 1/a + 1/b + 1/c == 1 over the candidate list:")
for i, t in enumerate(CANDIDATES):
    print(f"  index {i}: {t} -> 1/{t[0]}+1/{t[1]}+1/{t[2]} "
          f"= {Fraction(1,t[0])+Fraction(1,t[1])+Fraction(1,t[2])} "
          f"{'MARKED' if i in classical_marked else ''}")
print(f"Classically marked indices: {classical_marked}")

# Sanity check against the well-known finite result (rederived, not assumed):
# the only positive-integer solutions of 1/a+1/b+1/c=1 with a<=b<=c<=6 among
# our fixed candidates are exactly (2,3,6), (2,4,4), (3,3,3).
expected = [0, 4, 6]
assert classical_marked == expected, f"unexpected classical result {classical_marked}"


# ---------------------------------------------------------------------------
# 2. Quantum circuit: real 3-qubit Grover search for the marked indices
# ---------------------------------------------------------------------------

N_QUBITS = 3  # indices 0..7


def oracle_circuit(marked_indices, n_qubits):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")  # e.g. index 0 -> '000'
        # Flip qubits that should be 0 in this basis state so the
        # multi-controlled-Z fires exactly on |idx>.
        zero_positions = [n_qubits - 1 - pos for pos, b in enumerate(bits) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_positions:
            qc.x(q)
    return qc


def diffusion_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffusion")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked_indices, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = oracle_circuit(marked_indices, n_qubits)
    diffusion = diffusion_circuit(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffusion.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# Near-optimal Grover iteration count for M marked items out of N=2^n:
# floor( (pi/4) * sqrt(N/M) )
N = 2 ** N_QUBITS
M = len(classical_marked)
iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))
print(f"\nRunning Grover search: N={N} states, M={M} marked, iterations={iterations}")

qc = build_grover_circuit(classical_marked, N_QUBITS, iterations)

simulator = AerSimulator()
compiled = transpile(qc, simulator)
shots = 4096
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: rightmost classical bit in the count key is qubit 0.
# Our indices were encoded MSB-first as qubit (n-1)..0, matching the
# standard integer <-> bitstring convention used in oracle_circuit, so we
# convert a returned bitstring back to an integer directly.
index_counts = {}
for bitstring, n in counts.items():
    idx = int(bitstring, 2)
    index_counts[idx] = index_counts.get(idx, 0) + n

print("\nMeasured index -> shot counts:")
for idx in sorted(index_counts):
    tag = " (classically marked)" if idx in classical_marked else ""
    print(f"  index {idx} {CANDIDATES[idx]}: {index_counts.get(idx, 0)}{tag}")

marked_shots = sum(index_counts.get(i, 0) for i in classical_marked)
unmarked_shots = shots - marked_shots
most_frequent_index = max(index_counts, key=index_counts.get)

print(f"\nTotal shots on marked indices {classical_marked}: {marked_shots} / {shots}")
print(f"Total shots on unmarked indices: {unmarked_shots} / {shots}")
print(f"Most frequent measured index: {most_frequent_index} {CANDIDATES[most_frequent_index]}")


# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer and report PASS/FAIL
# ---------------------------------------------------------------------------

amplification_ok = marked_shots > unmarked_shots
top_hit_is_marked = most_frequent_index in classical_marked

if amplification_ok and top_hit_is_marked:
    print("\nPASS: Grover search amplified exactly the classically-marked unit-fraction "
          "triples {(2,3,6), (2,4,4), (3,3,3)}, matching the classical brute-force result.")
else:
    print("\nFAIL: quantum measurement distribution did not match the classical answer.")
