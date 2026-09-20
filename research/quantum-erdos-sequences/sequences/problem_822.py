"""
Erdos problem #822  (https://www.erdosproblems.com/822)

Erdos asked whether the set of integers representable as n + phi(n)
(n a positive integer, phi = Euler's totient function) has positive
lower density. This is exactly the sequence OEIS A121048: a(n) = n + phi(n).
(A155085, a(n) = n + sigma(n), is the companion sequence Erdos also asked
about, using the sum-of-divisors function instead of phi; not used below.)

Classical property tested here (finite, computable):

    Grover search over n in {1, ..., 16} (encoded as a 4-qubit index
    idx = n - 1, so idx in {0,...,15}) for the unique n with
    n + phi(n) == TARGET, where TARGET = 9 + phi(9) = 15.

The script first computes phi(n) and a(n) = n + phi(n) for n = 1..16
from first principles (trial coprimality test), and confirms in Python
that n = 9 is the *unique* solution to a(n) = 15 in this range. It then
builds a genuine Grover search circuit (4 qubits, an oracle built from
the classically-determined marked index, and the standard diffusion
operator) with the Grover-optimal number of iterations for a single
marked item out of 16, runs it on the ideal AerSimulator, and checks
that the most frequently measured basis state decodes to n = 9 -
i.e. that the quantum search recovers the same term of A121048 that
was found classically.
"""

from math import gcd, pi, floor

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the property, from first principles.
# ---------------------------------------------------------------------------

def euler_phi(n: int) -> int:
    """Euler's totient function computed by direct coprimality counting."""
    return sum(1 for k in range(1, n + 1) if gcd(n, k) == 1)


N = 16  # search space size -> 4 qubits (idx = n - 1, n = 1..16)
NUM_QUBITS = 4

a_values = {n: n + euler_phi(n) for n in range(1, N + 1)}

TARGET = a_values[9]  # = 9 + phi(9) = 9 + 6 = 15
assert TARGET == 15

solutions = [n for n in range(1, N + 1) if a_values[n] == TARGET]
assert solutions == [9], f"expected a unique solution n=9, got {solutions}"

CLASSICAL_N = solutions[0]
CLASSICAL_IDX = CLASSICAL_N - 1  # 0-based index Grover searches over
MARKED_BITSTRING = format(CLASSICAL_IDX, f"0{NUM_QUBITS}b")  # e.g. '1000'

print(f"n + phi(n) table (n=1..{N}): {a_values}")
print(f"TARGET = 9 + phi(9) = {TARGET}")
print(f"Unique classical solution: n = {CLASSICAL_N} "
      f"(index {CLASSICAL_IDX}, bitstring {MARKED_BITSTRING})")


# ---------------------------------------------------------------------------
# 2. Genuine Grover search circuit for the marked index.
# ---------------------------------------------------------------------------

def build_oracle(num_qubits: int, marked_bitstring: str) -> QuantumCircuit:
    """Phase oracle that flips the sign of |marked_bitstring>.

    Qiskit bit ordering is little-endian (qubit 0 is the least
    significant / rightmost bit), so we apply X gates on the qubits
    whose corresponding bit in the (big-endian) bitstring is 0, then a
    multi-controlled Z, then undo the X gates.
    """
    qc = QuantumCircuit(num_qubits, name="oracle")
    # bitstring[0] is the MSB -> corresponds to qubit (num_qubits-1)
    for i, bit in enumerate(reversed(marked_bitstring)):
        if bit == "0":
            qc.x(i)
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    for i, bit in enumerate(reversed(marked_bitstring)):
        if bit == "0":
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


oracle = build_oracle(NUM_QUBITS, MARKED_BITSTRING)
diffuser = build_diffuser(NUM_QUBITS)

# Grover-optimal iteration count for 1 marked item out of 2^NUM_QUBITS.
num_iterations = max(1, floor((pi / 4) * (2 ** (NUM_QUBITS / 2))))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(num_iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

print(f"Grover iterations used: {num_iterations}")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit returns bitstrings MSB-first as printed, matching our big-endian
# MARKED_BITSTRING convention directly.
most_common_bitstring = max(counts, key=counts.get)
most_common_idx = int(most_common_bitstring, 2)
most_common_n = most_common_idx + 1
most_common_count = counts[most_common_bitstring]
success_probability = most_common_count / shots

print(f"Measurement counts: {counts}")
print(f"Most frequent outcome: {most_common_bitstring} "
      f"(n = {most_common_n}), probability {success_probability:.3f}")

verified = (most_common_bitstring == MARKED_BITSTRING) and (most_common_n == CLASSICAL_N)

if verified:
    print(f"PASS: Grover search recovered n = {most_common_n}, "
          f"matching the classical solution to n + phi(n) = {TARGET} "
          f"(OEIS A121048).")
else:
    print(f"FAIL: Grover search returned n = {most_common_n}, "
          f"but the classical solution is n = {CLASSICAL_N}.")
