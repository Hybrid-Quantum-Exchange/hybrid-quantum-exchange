"""
Erdos problem #372 -- quantum-testable instance.

OEIS: A071870 -- numbers k such that gpf(k) > gpf(k+1) > gpf(k+2), where
gpf(n) is the greatest prime factor of n. (Erdos conjectured this sequence
is infinite; Balog (2001) proved it is.)

Classical property tested here (computed from first principles in this
script, not copied from OEIS): for k in the finite range 0..15 (a 4-qubit
search space, N = 16), which k satisfy gpf(k) > gpf(k+1) > gpf(k+2)?
gpf(0) and gpf(1) are undefined/have no prime factor, so k=0 and k=1 are
excluded from the domain by definition (gpf requires an integer >= 2).

We first compute the classical answer set exactly with a straightforward
greatest-prime-factor routine. We then build a genuine Grover search
circuit over the 4-qubit index register that marks exactly the classical
solution set via a multi-controlled-Z oracle (each solution index is one
computational basis state, so the oracle is a legitimate reflection about
those marked states -- not a shortcut around search). Grover's algorithm
then amplifies the marked amplitudes; we run the amplified circuit on the
ideal AerSimulator and check that measurement recovers the same solution
set with high probability, matching the classical computation.

Because the oracle marks precomputed classical index labels, the quantum
step being verified is genuinely the amplitude-amplification / search
step of Grover's algorithm (correct amplification of an oracle-marked
subset), not the arithmetic re-derivation of gpf itself -- that arithmetic
is intentionally done classically up front, exactly as Grover's algorithm
assumes a black-box oracle.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def gpf(n: int) -> int:
    """Greatest prime factor of n (n >= 2)."""
    if n < 2:
        raise ValueError("gpf is only defined for integers >= 2")
    largest = 1
    d = 2
    m = n
    while d * d <= m:
        while m % d == 0:
            largest = d
            m //= d
        d += 1
    if m > 1:
        largest = m
    return largest


def classical_solutions(n_min: int, n_max_exclusive: int):
    """k in [n_min, n_max_exclusive) with gpf(k) > gpf(k+1) > gpf(k+2)."""
    sols = []
    for k in range(n_min, n_max_exclusive):
        if k < 2 or k + 1 < 2 or k + 2 < 2:
            continue
        if gpf(k) > gpf(k + 1) > gpf(k + 2):
            sols.append(k)
    return sols


N_QUBITS = 4
N = 2 ** N_QUBITS  # search space size = 16, indices 0..15

solution_set = classical_solutions(0, N)
print(f"Classical search space: k in [0, {N})")
print(f"Classical solutions (gpf(k) > gpf(k+1) > gpf(k+2)): {solution_set}")

# Sanity check against the known start of A071870 (13, 14, 34, 37, ...):
# within range [0, 16) only 13 and 14 should appear.
expected_within_range = [k for k in [13, 14, 34, 37, 38, 43] if k < N]
assert solution_set == expected_within_range, (
    f"classical computation {solution_set} does not match expected "
    f"A071870 prefix restricted to range: {expected_within_range}"
)


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


n_marked = len(solution_set)
iterations = optimal_grover_iterations(N, n_marked)
print(f"Grover iterations used: {iterations} (marked={n_marked} out of N={N})")

oracle = build_oracle(N_QUBITS, solution_set)
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
marked_prob = sum(c for v, c in measured.items() if v in solution_set) / shots

print(f"Total probability mass on classically-marked states: {marked_prob:.3f}")

quantum_matches_classical = (
    measured_top_values == set(solution_set) and marked_prob > 0.8
)

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
