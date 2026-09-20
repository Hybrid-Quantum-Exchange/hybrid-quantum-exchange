"""
Erdos problem #45 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, as cloned at
/home/user/manman4/erdosproblems/data/problems.yaml, entry "number: '45'"):
    prize: no
    status: proved (Lean)
    oeis: ["possible"]
    tags: ["number theory", "unit fractions", "ramsey theory"]

LIMITATION, stated honestly up front: the dataset does not give a real OEIS
sequence id for problem #45 -- the oeis field literally contains the string
"possible", not an id such as "A000045". There is therefore no specific OEIS
sequence to build a membership/term test against. Rather than fabricate an
OEIS id or copy a value with no grounding, this script instead builds a
small, finite, genuinely computable problem drawn straight from the problem's
own tags ("unit fractions"): Egyptian-fraction (unit fraction) decompositions
of 1 into three distinct unit fractions,

    1/a + 1/b + 1/c = 1,  with integers 2 <= a < b < c <= N.

This is a classical, well-known finite Diophantine search (the only integer
solution with a<b<c is (a,b,c) = (2,3,6); this is elementary and is verified
below by brute force, not asserted) and it is exactly the kind of object
("unit fractions") the problem's own tags point at, even though it is not
tied to a specific OEIS id. This is presented as a best-effort finite
instance inspired by the problem's tags, NOT as a formalization of the open
mathematical content of problem #45 itself.

Classical step (done first, from first principles, in this script):
  Enumerate every triple 2 <= a < b < c <= N (N = 6 below, so the search
  space has C(5,3) = 10 candidate triples, indexable with 4 qubits) and
  check 1/a + 1/b + 1/c == 1 exactly using Python's Fraction class. This
  finds the unique classical answer (a, b, c) = (2, 3, 6).

Quantum step:
  A Grover search circuit is built over the 4-qubit index register spanning
  the 10 candidate triples (extended to 16 basis states; the 6 unused states
  are simply never marked). The oracle phase-flips exactly the index of the
  triple found classically to satisfy the equation; the diffuser amplifies
  it. The circuit is run on the ideal AerSimulator and the most frequently
  measured index is compared against the classically-found index.

PASS/FAIL: the script prints PASS if Grover's most-sampled outcome equals
the classically verified index of (2, 3, 6), FAIL otherwise.
"""

import math
from fractions import Fraction
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical stage: enumerate the search space and find the true answer.
# ---------------------------------------------------------------------------

N = 6  # 2 <= a < b < c <= N

triples = list(combinations(range(2, N + 1), 3))  # all (a,b,c), a<b<c<=N
num_triples = len(triples)
n_qubits = max(1, math.ceil(math.log2(num_triples)))  # index register width

assert num_triples <= 2 ** n_qubits

target_indices = []
for idx, (a, b, c) in enumerate(triples):
    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == 1:
        target_indices.append(idx)

# Classical ground truth, derived here, not assumed:
assert target_indices, "no unit-fraction solution found classically"
assert len(target_indices) == 1, "expected the unique solution (2,3,6)"
classical_index = target_indices[0]
classical_triple = triples[classical_index]
assert classical_triple == (2, 3, 6)

print(f"Search space: {num_triples} triples (a<b<c<={N}), {n_qubits}-qubit index register")
print(f"Classical answer: triple #{classical_index} = {classical_triple} "
      f"(1/{classical_triple[0]} + 1/{classical_triple[1]} + 1/{classical_triple[2]} = 1)")

# ---------------------------------------------------------------------------
# 2. Quantum stage: Grover search for classical_index among 2**n_qubits states.
# ---------------------------------------------------------------------------


def bits_of(value, width):
    return [(value >> i) & 1 for i in range(width)]


def apply_oracle(qc, marked_index, width):
    """Phase-flip the |marked_index> basis state."""
    bits = bits_of(marked_index, width)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    if width == 1:
        qc.z(0)
    else:
        qc.h(width - 1)
        qc.mcx(list(range(width - 1)), width - 1)
        qc.h(width - 1)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)


def apply_diffuser(qc, width):
    """Standard Grover diffuser (inversion about the mean)."""
    for i in range(width):
        qc.h(i)
        qc.x(i)
    if width == 1:
        qc.z(0)
    else:
        qc.h(width - 1)
        qc.mcx(list(range(width - 1)), width - 1)
        qc.h(width - 1)
    for i in range(width):
        qc.x(i)
        qc.h(i)


M = 2 ** n_qubits  # size of the (padded) search space
num_solutions = 1  # exactly one marked index (classical_index)

# Optimal number of Grover iterations for one marked item out of M.
iterations = max(1, round((math.pi / 4) * math.sqrt(M / num_solutions)))

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    apply_oracle(qc, classical_index, n_qubits)
    apply_diffuser(qc, n_qubits)
qc.measure(range(n_qubits), range(n_qubits))

sim = AerSimulator()
compiled = transpile(qc, sim)
job = sim.run(compiled, shots=2048)
result = job.result()
counts = result.get_counts()

# Qiskit's bitstring is written clbit[n-1] ... clbit[0], which already
# matches the register's standard big-endian-to-value reading since clbit i
# was measured from qubit i.
most_common_bitstring = max(counts, key=counts.get)
quantum_index = int(most_common_bitstring, 2)
quantum_triple = triples[quantum_index] if quantum_index < num_triples else None

print(f"Grover iterations: {iterations}")
print(f"Measurement counts (top 5): "
      f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Quantum most-likely index: {quantum_index} -> triple {quantum_triple}")

# ---------------------------------------------------------------------------
# 3. Compare quantum result against the classical answer.
# ---------------------------------------------------------------------------

verified = (quantum_index == classical_index) and (quantum_triple == classical_triple)

if verified:
    print("PASS")
else:
    print("FAIL")
