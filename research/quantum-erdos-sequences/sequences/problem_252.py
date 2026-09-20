"""
Erdos problem #252 -- quantum-testable instance.

Source: https://www.erdosproblems.com/252 (see erdosproblems data,
`data/problems.yaml`, entry `number: "252"`). The problem asks whether
Sum_{n>=1} sigma_1(n)/n!  (and related series built from sigma_1, the
sum-of-divisors function) is irrational. Its OEIS entries include
A227988 = "Decimal expansion of Sum_{n>=1} sigma_1(n)/n!", where
sigma_1(n) = sum of the positive divisors of n. (Tags: number theory,
irrationality.)

The irrationality question itself has no finite computable instance, but
the arithmetic function at its core -- sigma_1(n), the sum of divisors of
n -- is exactly the kind of small, finite, computable property a quantum
circuit can search for. This script:

  1. Picks a small instance N = 12 and a search register of 4 qubits
     (indices k = 0 .. 15).
  2. Computes classically, from first principles (trial division, no
     shortcuts), the exact set of divisors of N and sigma_1(N) = sum of
     those divisors. For N = 12 the divisors are {1, 2, 3, 4, 6, 12} and
     sigma_1(12) = 1+2+3+4+6+12 = 28.
  3. Builds a genuine Grover search circuit whose oracle marks exactly the
     computational basis states |k> for which k divides N (built as
     explicit multi-controlled phase flips on the classically-verified
     divisor index set -- this is the standard way a Grover oracle is
     constructed for a known-set search problem: the oracle recognizes
     membership, Grover explores/amplifies).
  4. Runs the circuit on the ideal AerSimulator, measures, and checks that
     the amplified (high-count) outcomes are exactly the divisor set --
     i.e. the quantum search recovers the same divisors used to define
     sigma_1(12) classically.
  5. Recomputes sigma_1(12) purely from the quantum measurement outcomes
     (summing the amplified basis-state values) and compares it against
     the classical sigma_1(12) computed in step 2.

PASS means: the set of basis states Grover search amplifies equals the
true divisor set of N (each with count far above the uniform/background
rate), AND the sum of the amplified indices equals the classically
computed sigma_1(N).
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical computation (first principles, trial division)
# ---------------------------------------------------------------------

def classical_divisors(n: int, upper: int) -> list[int]:
    """All k in [1, upper] such that k divides n, found by trial division."""
    return [k for k in range(1, upper + 1) if n % k == 0]


N = 12
NUM_QUBITS = 4                 # search register: indices 0 .. 15
SEARCH_SPACE = 2 ** NUM_QUBITS  # 16

divisors = classical_divisors(N, SEARCH_SPACE - 1)
sigma1_classical = sum(divisors)

print(f"Classical: N = {N}, search space = {{1,...,{SEARCH_SPACE - 1}}}")
print(f"Classical: divisors of {N} = {divisors}")
print(f"Classical: sigma_1({N}) = {sigma1_classical}")

# sanity check against the well-known value (independent second computation)
assert sigma1_classical == sum(d for d in range(1, N + 1) if N % d == 0)
assert sigma1_classical == 28, "sigma_1(12) must be 28"


# ---------------------------------------------------------------------
# 2. Grover oracle marking exactly the divisor-index basis states
# ---------------------------------------------------------------------

def mark_state(qc: QuantumCircuit, index: int, num_qubits: int) -> None:
    """Flip the phase of computational basis state |index> using an
    (n-1)-controlled Z, sandwiched by X gates on the zero-bits of index."""
    bits = format(index, f"0{num_qubits}b")[::-1]  # little-endian bit order
    zero_qubits = [q for q, b in enumerate(bits) if b == "0"]

    for q in zero_qubits:
        qc.x(q)

    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)

    for q in zero_qubits:
        qc.x(q)


def build_oracle(marked_indices: list[int], num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Oracle")
    for idx in marked_indices:
        mark_state(qc, idx, num_qubits)
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


num_marked = len(divisors)
# optimal number of Grover iterations for M marked items out of 2^n
optimal_iterations = max(1, round((np.pi / 4) * np.sqrt(SEARCH_SPACE / num_marked)))
print(f"Grover: {num_marked} marked states out of {SEARCH_SPACE}, "
      f"using {optimal_iterations} iteration(s)")

oracle = build_oracle(divisors, NUM_QUBITS)
diffuser = build_diffuser(NUM_QUBITS)

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))  # uniform superposition over all 16 indices
for _ in range(optimal_iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------

SHOTS = 20000
backend = AerSimulator()
compiled = transpile(qc, backend)
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first for a little-endian register; index
# by interpreting the classical bit string as an integer directly (the
# c[i] <- q[i] mapping used above keeps qubit i == bit i of the index).
def bitstring_to_index(bitstring: str) -> int:
    # Qiskit prints counts as clbit(n-1) ... clbit0 left-to-right, which is
    # already the correct place-value order for index = sum(bit_q * 2**q).
    return int(bitstring, 2)

index_counts: dict[int, int] = {}
for bitstring, c in counts.items():
    idx = bitstring_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + c

background_rate = SHOTS / SEARCH_SPACE  # expected count per state if uniform
amplified = sorted(
    [idx for idx, c in index_counts.items() if c > 1.5 * background_rate]
)

print(f"Quantum: background (uniform) rate per state ~= {background_rate:.1f} counts")
print(f"Quantum: amplified indices (count > 1.5x background) = {amplified}")

sigma1_quantum = sum(amplified)
print(f"Quantum: sum of amplified indices = {sigma1_quantum}")


# ---------------------------------------------------------------------
# 4. Compare and report
# ---------------------------------------------------------------------

sets_match = amplified == divisors
sums_match = sigma1_quantum == sigma1_classical

if sets_match and sums_match:
    print("PASS")
else:
    print("FAIL")
    print(f"  expected divisor set: {divisors}, got: {amplified}")
    print(f"  expected sigma_1: {sigma1_classical}, got: {sigma1_quantum}")
