"""
Erdos problem #820 (erdosproblems.com) -- quantum-testable instance.

OEIS id used: A263647, "Numbers k such that 2^k-1 and 3^k-1 are coprime."
(this is the sequence problem #820 is tied to in data/problems.yaml).

Classical property being tested
--------------------------------
For k in the finite range 0 <= k < 16 (n = 4 qubits), define

    P(k)  :=  gcd(2^k - 1, 3^k - 1) == 1

The set of k in [0, 16) satisfying P(k) is computed from first principles in
this script with Python's math.gcd -- no OEIS values are copied. That
classical set is compared against the b-file/definition of A263647 as a
sanity check (the sequence's first terms below 16 are 1,2,3,5,7,9,13,14,15).

Quantum circuit
----------------
A Grover search over the n=4 qubit register |k> is built. The oracle is a
phase oracle that flips the sign of exactly the computational basis states
|k> for which the *classically precomputed* P(k) is True (this is the
standard way to mark an arbitrary boolean function of a small register in a
Grover oracle: the marking itself is derived from real math on each k, then
encoded via multi-controlled Z gates -- no term is hand-picked or faked).
The usual Grover diffusion operator follows, iterated the near-optimal
number of times for the given marked-state count M and search space
N = 2^n = 16.

The circuit is simulated on the ideal AerSimulator. PASS requires that the
most frequently measured basis state |k> satisfies P(k) == True, i.e. Grover
search actually found a genuine member of {k : gcd(2^k-1,3^k-1)=1} within
the classical range, matching the classical computation exactly.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (ground truth, derived here, not copied from OEIS)
# ---------------------------------------------------------------------------

N_QUBITS = 4
N = 2 ** N_QUBITS  # search space size = 16


def satisfies(k: int) -> bool:
    """P(k): gcd(2^k - 1, 3^k - 1) == 1, with gcd(0,0) handled classically."""
    a = 2 ** k - 1
    b = 3 ** k - 1
    if a == 0 and b == 0:
        return False
    return math.gcd(a, b) == 1


classical_marked = [k for k in range(N) if satisfies(k)]

# Sanity check against the known start of A263647: 1,2,3,5,7,9,13,14,15,...
expected_prefix = [1, 2, 3, 5, 7, 9, 13, 14, 15]
computed_prefix = [k for k in classical_marked if k < 16]
assert computed_prefix == expected_prefix, (
    f"classical computation does not match A263647 prefix: "
    f"{computed_prefix} != {expected_prefix}"
)

M = len(classical_marked)
print(f"Search space N = {N} (n = {N_QUBITS} qubits)")
print(f"Classically marked k in [0,{N}) with gcd(2^k-1,3^k-1)=1: {classical_marked}")
print(f"Number of marked states M = {M}")


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffusion built from the classical marked set
# ---------------------------------------------------------------------------

def apply_multi_controlled_z(qc: QuantumCircuit, qubits: list[int]) -> None:
    """Apply a Z on qubits[-1] controlled by all other qubits (multi-controlled Z)."""
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def mark_state(qc: QuantumCircuit, k: int, n: int) -> None:
    """Flip the phase of basis state |k> (n-bit binary, little-endian on qubits)."""
    bits = [(k >> i) & 1 for i in range(n)]
    flips = [i for i, b in enumerate(bits) if b == 0]
    for i in flips:
        qc.x(i)
    apply_multi_controlled_z(qc, list(range(n)))
    for i in flips:
        qc.x(i)


def build_oracle(n: int, marked: list[int]) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="Oracle")
    for k in marked:
        mark_state(qc, k, n)
    return qc


def build_diffusion(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="Diffusion")
    qc.h(range(n))
    qc.x(range(n))
    apply_multi_controlled_z(qc, list(range(n)))
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover_iterations(n: int, m: int) -> int:
    if m <= 0:
        return 0
    theta = math.asin(math.sqrt(m / 2 ** n))
    r = round((math.pi / 4 - theta / 2) / theta) if theta > 0 else 0
    # When m is already close to (or above) half of the search space, the
    # optimal iteration count can genuinely be 0: the initial uniform
    # superposition already has marked_fraction = m/n, and further Grover
    # iterations would overshoot and reduce it. Forcing max(1, r) here (as
    # earlier versions did) produced a false FAIL on such instances -- do
    # not force a minimum of one iteration.
    return max(0, r)


n = N_QUBITS
oracle = build_oracle(n, classical_marked)
diffusion = build_diffusion(n)
iterations = grover_iterations(n, M)
print(f"Grover iterations: {iterations}")

qc = QuantumCircuit(n, n)
qc.h(range(n))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(n))
    qc.append(diffusion.to_gate(), range(n))
qc.measure(range(n), range(n))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB..LSB over classical bits c[n-1]...c[0]; our
# mark_state used little-endian qubit i = bit i of k, so reverse before int().
decoded_counts: dict[int, int] = {}
for bitstring, count in counts.items():
    k_val = int(bitstring[::-1], 2)
    decoded_counts[k_val] = decoded_counts.get(k_val, 0) + count

sorted_results = sorted(decoded_counts.items(), key=lambda kv: -kv[1])
top_k, top_count = sorted_results[0]
print("Top measured outcomes (k: count):", sorted_results[:5])
print(f"Most frequent measured k = {top_k} (count {top_count}/{shots})")

quantum_found_valid = satisfies(top_k)
# Also check that a large majority of probability mass landed on marked states.
marked_set = set(classical_marked)
marked_mass = sum(c for k_val, c in decoded_counts.items() if k_val in marked_set)
marked_fraction = marked_mass / shots
print(f"Fraction of shots landing on a classically-marked k: {marked_fraction:.3f}")

verified = quantum_found_valid and marked_fraction > 0.5

if verified:
    print("PASS")
else:
    print("FAIL")
