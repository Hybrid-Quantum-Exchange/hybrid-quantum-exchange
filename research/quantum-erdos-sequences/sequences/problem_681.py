"""
Erdos problem #681 -- quantum-testable instance.

OEIS: A389680.

Definition (A389680): a positive integer k belongs to the sequence iff there
is no integer i > 1 such that k + i is composite and the least prime factor
of k + i exceeds i^2. Erdos problem #681 asks whether this sequence is
finite (it is conjectured to have only finitely many, and only 68 terms are
currently known).

Classical property tested here (derived and checked in this script, not
copied from OEIS): membership of k in {1, ..., 32} in A389680, restricted to
witnesses i in {2, 3, 4, 5, 6}. Because the least prime factor of a
composite number k+i is always <= sqrt(k+i), the condition
"least prime factor of k+i exceeds i^2" forces i^2 < sqrt(k+i), i.e.
i < (k+i)^(1/4). For k in [1, 32] this means only i = 2 can possibly produce
a witness (i = 3 would need k+3 >= 3^8 = 6561, far outside the range; the
script still checks i up to 6 exhaustively as a safety margin, and finds no
extra witnesses). A direct classical search (done below, first-principles
trial division, no OEIS values copied) shows that within k = 1..32 the
*only* excluded integer is k = 23, witnessed by i = 2: 23 + 2 = 25 is
composite with least prime factor 5 > 2^2 = 4. Every other k in [1, 32] is a
genuine (small, unconditionally verified) member of A389680.

This is exactly the classic "one marked item among N" Grover-search shape:
search space {1, ..., 32} (5 qubits), oracle marks the unique non-member
k = 23 found by exhaustive classical search over the same finite window and
witness range used to build the oracle. We build a real Grover circuit
(uniform superposition -> phase oracle marking |23-1> -> diffusion,
repeated the optimal number of times for N=32, M=1 marked item) on
AerSimulator, and check that the circuit recovers the classically-computed
excluded index with high probability.

No qiskit/aer/numpy dependency beyond what is already installed.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS values copied).
# ---------------------------------------------------------------------------

def least_prime_factor(n: int):
    if n < 2:
        return None
    i = 2
    while i * i <= n:
        if n % i == 0:
            return i
        i += 1
    return n  # n itself is prime


def is_composite(n: int) -> bool:
    lpf = least_prime_factor(n)
    return lpf is not None and lpf < n


N = 32  # search space size: k = 1 .. 32
I_MAX = 6  # witness range i = 2 .. I_MAX (exhaustive; see docstring for why
           # i > 6 cannot matter in this range)


def excluded_from_a389680(k: int) -> bool:
    """True if k is excluded from A389680 by some witness i in [2, I_MAX]."""
    for i in range(2, I_MAX + 1):
        v = k + i
        if is_composite(v) and least_prime_factor(v) > i * i:
            return True
    return False


classical_excluded = [k for k in range(1, N + 1) if excluded_from_a389680(k)]

# Sanity check against the known mathematical fact quoted in the OEIS entry:
# 23 is the classic first excluded number (witness i=2, 23+2=25=5^2).
assert classical_excluded == [23], (
    f"classical search found {classical_excluded}, expected exactly [23]"
)

target_k = classical_excluded[0]
target_index = target_k - 1  # 0-indexed position in the N=32 search space
n_qubits = int(math.log2(N))
assert 2 ** n_qubits == N

print(f"Classical result: A389680 excludes exactly k={target_k} among 1..{N}")
print(f"Target computational basis index (k-1) = {target_index} "
      f"= {format(target_index, f'0{n_qubits}b')} in binary")


# ---------------------------------------------------------------------------
# 2. Grover search circuit marking the unique excluded index.
# ---------------------------------------------------------------------------

def build_oracle(n_qubits: int, marked_index: int) -> QuantumCircuit:
    """Phase oracle flipping the sign of |marked_index> only."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked_index, f"0{n_qubits}b")[::-1]  # little-endian
    zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
    if zero_qubits:
        qc.x(zero_qubits)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    if zero_qubits:
        qc.x(zero_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


M = 1  # exactly one marked item
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))

oracle = build_oracle(n_qubits, target_index)
diffuser = build_diffuser(n_qubits)
for _ in range(optimal_iterations):
    qc.append(oracle.to_gate(), range(n_qubits))
    qc.append(diffuser.to_gate(), range(n_qubits))

qc.measure(range(n_qubits), range(n_qubits))

print(f"Grover circuit: {n_qubits} qubits, {optimal_iterations} iteration(s)")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 2048
transpiled = transpile(qc, backend)
job = backend.run(transpiled, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-left, matching our little-endian qubit
# indexing when read normally (qubit 0 = rightmost char).
best_bitstring = max(counts, key=counts.get)
best_index = int(best_bitstring, 2)
best_probability = counts[best_bitstring] / shots

print(f"Measurement counts (top 5): "
      f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Most frequent outcome: {best_bitstring} -> index {best_index} "
      f"(k={best_index + 1}), probability {best_probability:.3f}")

verified = (best_index == target_index) and (best_probability > 0.5)

if verified:
    print(f"PASS: Grover search recovered k={best_index + 1} as the unique "
          f"element of {{1,...,{N}}} excluded from A389680, matching the "
          f"classical computation (k={target_k}).")
else:
    print(f"FAIL: quantum result (k={best_index + 1}, "
          f"p={best_probability:.3f}) did not match classical answer "
          f"(k={target_k}) with sufficient confidence.")

assert verified
