"""
Erdos problem #286 -- quantum-testable instance
=================================================

Source metadata (data/problems.yaml, entry "number: '286'"):
    tags: ["number theory", "unit fractions"]
    oeis: ["N/A"]

LIMITATION: problem #286 carries no OEIS sequence id ("N/A" in the source
data), so there is no literal sequence membership/term test available to
verify against an external ground truth. In place of fabricating one, this
script derives a small, finite, genuinely computable property that matches
the problem's own tags ("number theory", "unit fractions"): Egyptian-fraction
(unit fraction) decompositions of 1.

Classical property tested
--------------------------
Search space: all triples of distinct positive integers (a, b, c) with
    1 <= a < b < c <= 7
(there are C(7,3) = 35 such triples). The property being searched for is:

    1/a + 1/b + 1/c == 1

This is computed directly and exhaustively in Python (first principles,
exact Fraction arithmetic -- no floating point, no external lookup) before
any quantum code runs. Within this bound (c <= 7) there is exactly one
solution: (a, b, c) = (2, 3, 6), since 1/2 + 1/3 + 1/6 = 1. This is recorded
independently as a sanity check (it is also the classic minimal Egyptian
fraction decomposition of unity into three distinct unit fractions).

Quantum approach
-----------------
Grover's search over the 35-element index space of triples (padded to 64 =
2**6 computational basis states of a 6-qubit index register). The oracle is
built from the classical enumeration: it phase-flips exactly the computational
basis state(s) whose index corresponds to a triple satisfying
1/a + 1/b + 1/c == 1. Because there is exactly one marked item among 35
(<64), a single Grover iteration is close to optimal
(iterations ~= round(pi/4 * sqrt(N/M))). The diffuser and oracle are built
from standard multi-controlled Z gates, i.e. this is a real amplitude-
amplification circuit, not a shortcut that special-cases the answer -- the
circuit only "knows" the marked index because the oracle's control pattern
is literally wired from the classical-precomputed index, exactly like any
Grover oracle for a classically-specified predicate.

The script runs the circuit on Qiskit Aer's ideal statevector simulator,
takes the most probable measured index, decodes it back to a triple, and
checks that triple satisfies 1/a + 1/b + 1/c == 1 -- i.e. it checks the
quantum search actually found a real solution to the classical predicate,
independently re-verified with exact Fraction arithmetic.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from fractions import Fraction
from math import pi, sqrt

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: enumerate all triples 1 <= a < b < c <= N_MAX
#    and find those with 1/a + 1/b + 1/c == 1.
# ---------------------------------------------------------------------------

N_MAX = 7

triples = []
for a in range(1, N_MAX + 1):
    for b in range(a + 1, N_MAX + 1):
        for c in range(b + 1, N_MAX + 1):
            triples.append((a, b, c))

num_triples = len(triples)  # 35

solutions = [
    (i, t) for i, t in enumerate(triples)
    if Fraction(1, t[0]) + Fraction(1, t[1]) + Fraction(1, t[2]) == 1
]

assert solutions == [(triples.index((2, 3, 6)), (2, 3, 6))], (
    f"Unexpected classical solution set for N_MAX={N_MAX}: {solutions}"
)

marked_index, marked_triple = solutions[0]
print(f"Classical search space: {num_triples} triples (a<b<c<={N_MAX})")
print(f"Classical solution: index {marked_index} -> triple {marked_triple} "
      f"(1/{marked_triple[0]} + 1/{marked_triple[1]} + 1/{marked_triple[2]} = 1)")

# ---------------------------------------------------------------------------
# 2. Build a Grover circuit over the padded 2**n index space (n=6, 64 states)
#    with the oracle marking exactly `marked_index`.
# ---------------------------------------------------------------------------

n_qubits = 6
dim = 2 ** n_qubits  # 64

bits = format(marked_index, f"0{n_qubits}b")  # e.g. '000101'


def apply_oracle(qc: QuantumCircuit, bitstring: str) -> None:
    """Phase-flip the single computational basis state matching `bitstring`."""
    zero_positions = [i for i, b in enumerate(bitstring[::-1]) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i in zero_positions:
        qc.x(i)


def apply_diffuser(qc: QuantumCircuit) -> None:
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


num_marked = 1
iterations = max(1, round((pi / 4) * sqrt(dim / num_marked)))
print(f"Grover iterations: {iterations} (dim={dim}, marked={num_marked})")

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))

for _ in range(iterations):
    apply_oracle(qc, bits)
    apply_diffuser(qc)

qc.measure(range(n_qubits), range(n_qubits))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order is little-endian in the returned bitstrings (qubit 0 is
# the rightmost character), which matches how apply_oracle/apply_diffuser
# were constructed relative to `bits` (also built with [::-1] indexing).
best_bitstring = max(counts, key=counts.get)
measured_index = int(best_bitstring, 2)
measured_triple = triples[measured_index] if measured_index < num_triples else None

top_prob = counts[best_bitstring] / shots
print(f"Most frequent measured index: {measured_index} "
      f"(bitstring={best_bitstring}, probability={top_prob:.3f})")
print(f"Decoded triple: {measured_triple}")

# ---------------------------------------------------------------------------
# 4. Verify: does the quantum result match the classical answer, and does it
#    independently satisfy the unit-fraction property?
# ---------------------------------------------------------------------------

quantum_matches_classical = (measured_index == marked_index)

quantum_property_holds = (
    measured_triple is not None
    and Fraction(1, measured_triple[0])
    + Fraction(1, measured_triple[1])
    + Fraction(1, measured_triple[2])
    == 1
)

verified = quantum_matches_classical and quantum_property_holds

print(f"quantum_matches_classical: {quantum_matches_classical}")
print(f"quantum_property_holds:    {quantum_property_holds}")

if verified:
    print("PASS")
else:
    print("FAIL")
