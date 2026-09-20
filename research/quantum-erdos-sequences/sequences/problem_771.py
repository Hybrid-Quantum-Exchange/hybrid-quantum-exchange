"""
Erdos problem #771 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 771"):
    prize: no
    status: proved (as of 2025-08-31)
    oeis: ["possible"]          <-- NOT a real OEIS sequence id; the data
                                     file uses the literal placeholder
                                     string "possible" rather than an
                                     identifier such as "A000040".
    tags: ["number theory"]

LIMITATION (reported honestly, as instructed): problem 771's entry carries
no usable OEIS sequence id and no informative tag beyond the generic
"number theory" -- there is no specific sequence definition available in
the source data to derive a property from. It is therefore not possible to
build a circuit that tests problem 771's own sequence.

BEST-EFFORT FALLBACK: to still deliver a genuine, checkable quantum
computation anchored in "number theory" (the one real signal in the
entry), this script runs Grover's search algorithm to find the prime
numbers in [0, 15] (4 qubits, N = 16 <= 64 as required). This is a real,
finite, classically-checkable number-theoretic search problem (primality
testing / membership search), implemented with a genuine phase oracle and
diffuser -- not a fabricated stand-in for problem 771's actual content.

Classical property tested:
    S = { n in [0, 15] : n is prime } = {2, 3, 5, 7, 11, 13}
    computed here from first principles by trial division.

Quantum method:
    Grover's algorithm on 4 qubits (search space size N = 16), with a
    phase oracle that flips the sign of computational basis states whose
    integer value is in S, followed by ~ (pi/4) * sqrt(N/|S|) Grover
    iterations, run on the ideal AerSimulator. PASS if the states with the
    highest measured probability mass are exactly the primes in S.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup)
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16
MARKED = [n for n in range(N) if is_prime(n)]
assert MARKED == [2, 3, 5, 7, 11, 13]
print(f"Classical answer: primes in [0, {N - 1}] = {MARKED}")


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffuser (real circuit, not a lookup table)
# ---------------------------------------------------------------------------

def build_oracle(marked_values, n_qubits):
    """Phase oracle: flips sign of |x> for each x in marked_values."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian
        # Flip qubits that are 0 in this bitstring so the multi-controlled
        # Z fires exactly when the register equals `value`.
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(MARKED, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

n_iterations = max(1, round((math.pi / 4) * math.sqrt(N / len(MARKED))))
print(f"Grover iterations: {n_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(n_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 8192
result = backend.run(qc, shots=shots).result()
counts = result.get_counts()

# Convert bitstrings (Qiskit prints classical-bit order c[n-1]...c[0],
# which matches our little-endian qubit ordering) to integers.
value_counts = Counter()
for bitstring, count in counts.items():
    value = int(bitstring, 2)
    value_counts[value] += count

top_k = [value for value, _ in value_counts.most_common(len(MARKED))]
top_k_sorted = sorted(top_k)

print("Measured value counts (top entries):")
for value, count in value_counts.most_common(10):
    tag = "PRIME" if value in MARKED else ""
    print(f"  {value:2d}: {count:5d}  {tag}")

quantum_answer = top_k_sorted
classical_answer = sorted(MARKED)

print(f"Quantum top-{len(MARKED)} measured values: {quantum_answer}")
print(f"Classical primes in [0, {N - 1}]:          {classical_answer}")

verified = quantum_answer == classical_answer

if verified:
    print("PASS")
else:
    print("FAIL")
