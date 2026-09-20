"""
Erdos problem #374 -- quantum-testable instance.

OEIS: A388851 -- numbers c such that a! * b! * c! is a perfect square for
some 1 <= a < b < c. (One of four OEIS sequences listed for this problem;
A388851 is the one used here because it is a small, finite, exactly
computable membership property.)

Classical property tested here (computed from first principles in this
script, not copied from OEIS): fix c = 4, the smallest term of A388851.
Enumerate every pair (a, b) with 1 <= a < b < c = 4, i.e. (a, b) in
{(1,2), (1,3), (2,3)}, and determine for which pair(s) a! * b! * c! is a
perfect square. There are exactly 3 candidate pairs, which we index by a
2-qubit register (search space N = 4; index 3 is unused/never marked
since there are only 3 pairs). We verify classically that c = 4 is
genuinely in A388851, i.e. that at least one such pair exists, and
identify exactly which index/pairs work using ordinary integer
arithmetic (factorial + integer square root), with no reference to any
OEIS b-file value.

We then build a genuine Grover search circuit over the 2-qubit index
register that marks exactly the classically-determined solution
index/indices via a multi-controlled-Z oracle (each solution index is
one computational basis state, so the oracle is a legitimate reflection
about those marked states -- not a shortcut around search). Grover's
algorithm then amplifies the marked amplitude(s); we run the amplified
circuit on the ideal AerSimulator and check that measurement recovers
the same solution index with high probability, matching the classical
computation.

Because the oracle marks a precomputed classical index label, the
quantum step being verified is genuinely the amplitude-amplification /
search step of Grover's algorithm (correct amplification of an
oracle-marked subset), not the arithmetic re-derivation of the
perfect-square test itself -- that arithmetic is intentionally done
classically up front, exactly as Grover's algorithm assumes a black-box
oracle.
"""

import math
from math import factorial, isqrt

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


C = 4  # smallest term of A388851, verified below

# All pairs (a, b) with 1 <= a < b < C, enumerated in a fixed order so we
# can index them with a 2-qubit register.
pairs = [(a, b) for a in range(1, C) for b in range(a + 1, C)]
print(f"c = {C}, candidate pairs (a, b) with 1 <= a < b < c: {pairs}")

c_fact = factorial(C)
solution_indices = []
for idx, (a, b) in enumerate(pairs):
    product = factorial(a) * factorial(b) * c_fact
    square = is_perfect_square(product)
    print(
        f"  index {idx}: (a,b)=({a},{b})  a!b!c! = {product}  "
        f"perfect square: {square}"
    )
    if square:
        solution_indices.append(idx)

# Classical ground truth: c = 4 is in A388851 iff at least one pair works.
assert len(solution_indices) > 0, (
    "c=4 should be in A388851 (OEIS first terms begin 4, 6, 8, ...); "
    "no witnessing pair (a,b) found -- classical computation disagrees "
    "with the known sequence membership."
)
# Cross-check the specific witness against hand computation:
# (a,b) = (1,3): 1! * 3! * 4! = 1 * 6 * 24 = 144 = 12^2.
assert solution_indices == [1], (
    f"expected the unique witness to be index 1 (a,b)=(1,3), got {solution_indices}"
)
print(f"Classical solution index/indices (marking c={C} in A388851): {solution_indices}")


N_QUBITS = 2
N = 2 ** N_QUBITS  # search space size = 4; indices 0..3 (index 3 unused)


def apply_multi_controlled_z(qc: QuantumCircuit, qubits, value: int, n_qubits: int):
    """Flip the phase of exactly the computational basis state |value>."""
    bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
    flip_qubits = [qubits[i] for i, b in enumerate(bits) if b == "0"]
    for q in flip_qubits:
        qc.x(q)
    if n_qubits == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in flip_qubits:
        qc.x(q)


def build_oracle(n_qubits: int, marked_values):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    qubits = list(range(n_qubits))
    for v in marked_values:
        apply_multi_controlled_z(qc, qubits, v, n_qubits)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qubits = list(range(n_qubits))
    qc.h(qubits)
    qc.x(qubits)
    if n_qubits == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


def optimal_grover_iterations(n_items: int, n_marked: int) -> int:
    theta = math.asin(math.sqrt(n_marked / n_items))
    r = round((math.pi / (4 * theta)) - 0.5)
    return max(1, r)


n_marked = len(solution_indices)
iterations = optimal_grover_iterations(N, n_marked)
print(f"Grover iterations used: {iterations} (marked={n_marked} out of N={N})")

oracle = build_oracle(N_QUBITS, solution_indices)
diffuser = build_diffuser(N_QUBITS)

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N_QUBITS))
    qc.append(diffuser.to_instruction(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-register bitstrings are printed MSB-first (qubit n-1 .. 0),
# and match the little-endian qubit ordering used in apply_multi_controlled_z.
measured = {int(bitstring, 2): c for bitstring, c in counts.items()}

top_k = sorted(measured.items(), key=lambda kv: -kv[1])[: max(n_marked, 1) + 2]
print("Top measured outcomes (value: count):")
for value, count in top_k:
    print(f"  {value}: {count} ({count / shots:.3f})")

measured_top_values = {v for v, _ in top_k[:n_marked]}
marked_prob = sum(c for v, c in measured.items() if v in solution_indices) / shots

print(f"Total probability mass on classically-marked states: {marked_prob:.3f}")

quantum_matches_classical = (
    measured_top_values == set(solution_indices) and marked_prob > 0.8
)

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
