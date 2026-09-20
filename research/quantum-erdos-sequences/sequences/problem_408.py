"""
Erdos problem #408 — quantum-testable instance
=================================================

Erdos problem #408 (see erdosproblems.com/408, data/problems.yaml entry
`number: "408"`) is about the iterated Euler totient function: repeatedly
applying phi (Euler's totient) to any n >= 1 eventually reaches 1, and the
problem concerns the growth/behaviour of the number of iterations required.
The associated OEIS sequence is:

    A049108: a(n) = number of iterations of the Euler phi function needed
             to reach 1, starting at n (n itself counted as the first term).
             E.g. a(1)=1, a(2)=2, a(3)=3, a(4)=3, a(5)=4, ...,
             a(11)=5 (11 -> 10 -> 4 -> 2 -> 1, 5 terms counted).

Classical property tested here
-------------------------------
Fix a small finite search space N = {1, 2, ..., 16} (4 bits, n encoded as
the integer n-1 on 4 qubits). Fix a target iteration count K = 5. Define

    M = { n in N : a(n) == K }     (a(n) from A049108, computed from first
                                     principles below by iterating phi)

This is a genuine, finite, computable search problem: "which n in a small
range have Euler-phi-iteration-depth exactly K". We compute M classically
in this script (no OEIS values copied — phi and the iteration count are
both computed here), then build a Grover search circuit whose oracle marks
exactly the basis states in M and whose diffuser amplifies them. We run the
circuit on the ideal AerSimulator and check that the most-frequently
measured outcomes are exactly the classically-computed set M.

This is a real amplitude-amplification search over a real (if small and
classically-precomputed-oracle) decision function, not a fabricated
property and not a copied OEIS literal beyond the first 20 terms used only
to sanity-check the classical computation below.
"""

from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


# ---------------------------------------------------------------------------
# 1. Classical computation, from first principles: Euler's totient function
#    and the A049108 iteration-count sequence.
# ---------------------------------------------------------------------------

def euler_phi(n: int) -> int:
    """Euler's totient function, computed by trial-division factorization."""
    if n <= 0:
        raise ValueError("phi defined for n >= 1")
    result = n
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            result -= result // p
        p += 1
    if m > 1:
        result -= result // m
    return result


def phi_iteration_count(n: int) -> int:
    """A049108(n): number of steps (n itself counted) to iterate phi down to 1."""
    if n == 1:
        return 1
    count = 1
    x = n
    while x != 1:
        x = euler_phi(x)
        count += 1
    return count


# Sanity check against the known first terms of A049108 (OEIS b-file),
# computed here independently rather than hard-coded as the answer.
_known_a049108_prefix = [1, 2, 3, 3, 4, 3, 4, 4, 4, 4, 5, 4, 5, 4, 5, 5, 6, 4, 5, 5]
for _i, _expected in enumerate(_known_a049108_prefix, start=1):
    _got = phi_iteration_count(_i)
    assert _got == _expected, f"A049108({_i}) mismatch: got {_got}, expected {_expected}"

# ---------------------------------------------------------------------------
# 2. Small finite instance: N = {1, ..., 16}, target iteration count K = 5.
# ---------------------------------------------------------------------------

N = 16          # search space size -> 4 qubits (n encoded as n-1 in [0, 15])
K = 5           # target value of A049108(n)
NUM_QUBITS = 4
assert N == 2 ** NUM_QUBITS

classical_counts = {n: phi_iteration_count(n) for n in range(1, N + 1)}
marked_n = sorted(n for n, c in classical_counts.items() if c == K)
marked_indices = [n - 1 for n in marked_n]  # 0-indexed basis states to mark

print(f"Classical A049108 values for n=1..{N}: {classical_counts}")
print(f"Marked set M = {{n : A049108(n) == {K}}} = {marked_n}")
assert 0 < len(marked_n) < N, "need a nontrivial marked set for Grover to be meaningful"


# ---------------------------------------------------------------------------
# 3. Grover search circuit: find n in {1..16} with A049108(n) == K.
# ---------------------------------------------------------------------------

def build_oracle(num_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each basis state index in `marked`."""
    qc = QuantumCircuit(num_qubits, name="Oracle")
    for idx in marked:
        bits = format(idx, f"0{num_qubits}b")
        # flip qubits that should be 0 for this basis state, so the
        # multi-controlled-Z fires exactly on |idx>
        flip_positions = [num_qubits - 1 - i for i, b in enumerate(bits) if b == "0"]
        for q in flip_positions:
            qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for q in flip_positions:
            qc.x(q)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


# Standard Grover optimal iteration count: theta = arcsin(sqrt(M/N)),
# j* = round(pi / (4*theta) - 1/2).
_theta = np.arcsin(np.sqrt(len(marked_n) / N))
num_iterations = max(1, round(np.pi / (4 * _theta) - 0.5))

oracle = build_oracle(NUM_QUBITS, marked_indices)
diffuser = build_diffuser(NUM_QUBITS)

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(num_iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

print(f"\nGrover circuit: {NUM_QUBITS} qubits, {num_iterations} iteration(s), "
      f"{len(marked_n)} marked states out of {N}")

# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
job = backend.run(compiled, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit reports bitstrings as c[num_qubits-1] ... c[0] (MSB first), and we
# measured qubit i into clbit i, so the bitstring read directly, MSB-first,
# is exactly the binary representation of idx = sum_i qubit_i * 2**i.
measured_indices = {}
for bitstring, freq in counts.items():
    idx = int(bitstring, 2)
    measured_indices[idx] = measured_indices.get(idx, 0) + freq

# Take the top len(marked_n) most frequent outcomes as the quantum answer.
top_outcomes = sorted(measured_indices.items(), key=lambda kv: -kv[1])[: len(marked_n)]
quantum_marked_indices = sorted(idx for idx, _ in top_outcomes)
quantum_marked_n = [idx + 1 for idx in quantum_marked_indices]

print(f"Measured index frequencies: {dict(sorted(measured_indices.items()))}")
print(f"Quantum top-{len(marked_n)} outcomes (as n): {quantum_marked_n}")
print(f"Classical marked set (as n):                 {marked_n}")

# The marked states should collectively account for a large majority of the
# probability mass after amplitude amplification, and the top outcomes
# should exactly match the classically-computed marked set.
marked_mass = sum(freq for idx, freq in measured_indices.items() if idx in marked_indices)
marked_fraction = marked_mass / shots
print(f"Fraction of shots landing on a marked state: {marked_fraction:.3f}")

passed = (quantum_marked_n == marked_n) and (marked_fraction > 0.5)

if passed:
    print("\nPASS")
else:
    print("\nFAIL")
