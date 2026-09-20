"""
Erdos problem #314 (unit fractions / Egyptian-fraction number theory,
tags: ["number theory", "unit fractions"]).

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: '314'")
lists oeis: ["N/A"] -- problem 314 has NO associated OEIS sequence. Per the task
instructions, in that situation this script gives its best honest attempt at a
genuine small quantum computation tied to the problem's *tags* rather than to
an OEIS sequence, and states the limitation plainly:

    LIMITATION: there is no OEIS id for this problem, so there is no
    "sequence" to make quantum-testable in the literal sense the library
    otherwise uses. What follows instead targets the "unit fractions" tag
    directly: an Egyptian-fraction (unit-fraction decomposition) search,
    which is real, finite, and classically checkable, in the spirit of the
    unit-fraction number theory this problem concerns (e.g. Erdos-Straus-type
    decompositions of 4/n into three unit fractions).

Classical property being tested (computed here, not looked up)
----------------------------------------------------------------
Fix n = 5 and two of the three unit-fraction denominators, a = 2 and b = 5.
Erdos-Straus asks whether 4/n can always be written as 1/a + 1/b + 1/c for
positive integers a, b, c. For n = 5 with a = 2, b = 5:

    4/5 - 1/2 - 1/5 = 1/c   =>   c = 10   (an exact integer solution)

We verify this by brute-force search over ALL c in the small finite range
[1, 16] using exact rational arithmetic (Python's fractions module), which
is the classical ground truth. The unique c in [1, 16] satisfying
1/2 + 1/5 + 1/c = 4/5 is c = 10.

Quantum circuit
----------------
A genuine Grover search over the 4-qubit space {0, 1, ..., 15} (representing
candidate values of c, with basis state |c-1> standing for c). The oracle is
a diagonal phase oracle built directly from the classically-verified marked
set (a single marked index, c = 10 -> index 9), and the diffuser is the
standard Grover diffusion operator. This is unstructured (Grover) search:
the quadratic-speedup search primitive applies unchanged whether the marked
set has one element or many, and the marking here is exactly the unit-
fraction equation's solution set for this finite instance, not an arbitrary
label.

Pass criterion: after running one Grover iteration (optimal for N=16, one
marked item: floor(pi/4 * sqrt(16)) = 3 iterations) on AerSimulator, the
most frequently measured basis state must decode to c = 10, matching the
classical brute-force answer.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from fractions import Fraction
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force the unit-fraction equation.
# ---------------------------------------------------------------------------

def classical_solution(n: int, a: int, b: int, c_max: int):
    """Return the set of c in [1, c_max] with 1/a + 1/b + 1/c == 4/n."""
    target = Fraction(4, n)
    remainder = target - Fraction(1, a) - Fraction(1, b)
    solutions = []
    for c in range(1, c_max + 1):
        if Fraction(1, c) == remainder:
            solutions.append(c)
    return solutions, remainder


N = 5
A = 2
B = 5
C_MAX = 16  # search space size = 16 = 2^4, so 4 qubits

marked_c_values, remainder = classical_solution(N, A, B, C_MAX)
assert remainder > 0, "remainder must be a positive unit fraction target"
assert len(marked_c_values) == 1, (
    f"expected exactly one solution in range, found {marked_c_values}"
)
classical_c = marked_c_values[0]
assert classical_c == 10, f"expected c=10 from first-principles derivation, got {classical_c}"

# Sanity-check with exact rational arithmetic once more, independently.
check = Fraction(1, A) + Fraction(1, B) + Fraction(1, classical_c)
assert check == Fraction(4, N), f"classical check failed: {check} != 4/{N}"

marked_index = classical_c - 1  # basis state |c-1> encodes candidate c
n_qubits = int(math.log2(C_MAX))
assert 2 ** n_qubits == C_MAX

print(f"Classical ground truth: unique c in [1,{C_MAX}] solving "
      f"1/{A} + 1/{B} + 1/c = 4/{N} is c = {classical_c} "
      f"(marked basis index {marked_index})")


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffuser built from the marked index.
# ---------------------------------------------------------------------------

def bits_of(index: int, width: int):
    return [(index >> i) & 1 for i in range(width)]


def build_oracle(marked_index: int, width: int) -> QuantumCircuit:
    """Phase-flip the single computational basis state |marked_index>."""
    qc = QuantumCircuit(width, name="oracle")
    bits = bits_of(marked_index, width)
    # Flip qubits that should be 0 in the marked index so the multi-controlled
    # Z fires exactly on |marked_index>.
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
    return qc


def build_diffuser(width: int) -> QuantumCircuit:
    qc = QuantumCircuit(width, name="diffuser")
    qc.h(range(width))
    qc.x(range(width))
    if width == 1:
        qc.z(0)
    else:
        qc.h(width - 1)
        qc.mcx(list(range(width - 1)), width - 1)
        qc.h(width - 1)
    qc.x(range(width))
    qc.h(range(width))
    return qc


num_marked = 1
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(C_MAX / num_marked)))

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))

oracle = build_oracle(marked_index, n_qubits)
diffuser = build_diffuser(n_qubits)

for _ in range(optimal_iterations):
    qc.append(oracle.to_gate(), range(n_qubits))
    qc.append(diffuser.to_gate(), range(n_qubits))

qc.measure(range(n_qubits), range(n_qubits))

print(f"Grover circuit: {n_qubits} qubits, {optimal_iterations} iterations, "
      f"searching {C_MAX} candidates for c with a={A}, b={B}, n={N}")


# ---------------------------------------------------------------------------
# 3. Run on AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: rightmost classical bit is qubit 0, so reverse to get
# our little-endian index convention back.
def bitstring_to_index(bitstring: str) -> int:
    return int(bitstring[::-1], 2)

best_bitstring = max(counts, key=counts.get)
best_index = bitstring_to_index(best_bitstring)
best_c = best_index + 1
best_prob = counts[best_bitstring] / shots

print(f"Most frequent measurement: bitstring={best_bitstring} -> "
      f"index={best_index} -> c={best_c} (probability {best_prob:.3f} over {shots} shots)")

verified = (best_c == classical_c) and (best_prob > 0.5)

if verified:
    print("PASS: Grover search on AerSimulator recovered the classically "
          f"verified unit-fraction solution c={classical_c} for "
          f"1/{A} + 1/{B} + 1/c = 4/{N}.")
else:
    print("FAIL: quantum result did not match the classical ground truth.")

assert verified
