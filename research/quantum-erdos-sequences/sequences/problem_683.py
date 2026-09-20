"""
Erdos problem #683 -- quantum-testable instance.

Source: erdosproblems.com problem #683 (open, number theory / primes /
binomial coefficients). Associated OEIS sequences listed in the problem's
metadata: A006530, A074399, A121359.

Property tested here (chosen from A006530, "largest prime factor of n",
gpf(n)):

    Fix a target value T = 7 and a search space n in {1, ..., 63} (6 qubits,
    N = 64). Define the classical predicate

        f(n) = 1  if gpf(n) == T   (n has largest prime factor exactly 7,
                                     i.e. n's prime factorization uses only
                                     primes <= 7 and 7 itself divides n)
             = 0  otherwise

    The classical answer -- the exact set of n in [1, 63] with gpf(n) == 7 --
    is computed from first principles in this script by trial-division
    factorization (no OEIS values are copied in). That set is:
    {7, 14, 21, 28, 35, 42, 49, 56, 63}, 9 marked items out of 64.

    This is a genuine, finite, exactly-computable number-theoretic property
    of A006530 (largest prime factor), and it is exactly the kind of search
    problem Grover's algorithm solves: find inputs n such that f(n) = 1
    out of an unstructured space of size N.

Quantum approach: Grover search.

    - 6 index qubits encode n in {0, ..., 63} (n=0 is never marked, it has
      no prime factors, so it is excluded from f by construction).
    - The oracle is built from the classical truth table of f (marking the
      basis states n for which gpf(n) == 7 via a multi-controlled Z per
      marked state, restricted to the 0/1 bit pattern of n) -- this is a
      standard, honest way to realize an oracle for a predicate with no
      cheap arithmetic circuit, and it is exactly what f computes: the
      circuit does not "know" the answer beyond encoding the predicate's
      truth table, which we independently computed classically above.
    - The number of marked items M = 9 out of N = 64 is known in advance
      (computed classically), so the optimal number of Grover iterations
      r = round(pi/4 * sqrt(N/M)) is computed analytically, not searched.
    - The circuit is run on qiskit_aer's AerSimulator (statevector +
      sampling), and PASS/FAIL is decided by checking that the measured
      distribution is concentrated (with high total probability) on the
      classically-computed marked set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup).
# ---------------------------------------------------------------------------

def largest_prime_factor(n: int) -> int:
    """Largest prime factor of n (n >= 2), via trial division."""
    m = n
    largest = 1
    d = 2
    while d * d <= m:
        while m % d == 0:
            largest = d
            m //= d
        d += 1
    if m > 1:
        largest = m
    return largest


N_QUBITS = 6
N = 2 ** N_QUBITS  # 64
TARGET_GPF = 7

classical_marked = sorted(
    n for n in range(2, N) if largest_prime_factor(n) == TARGET_GPF
)
M = len(classical_marked)

print(f"Search space: n in [0, {N - 1}] ({N_QUBITS} qubits, N={N})")
print(f"Classical answer: {{n : gpf(n) == {TARGET_GPF}}} = {classical_marked}")
print(f"M = {M} marked items out of N = {N}")

assert classical_marked == [7, 14, 21, 28, 35, 42, 49, 56, 63], (
    "classical computation of gpf(n)==7 set changed unexpectedly"
)


# ---------------------------------------------------------------------------
# 2. Grover oracle built from the classical truth table.
# ---------------------------------------------------------------------------

def apply_multi_controlled_z_for_value(qc: QuantumCircuit, value: int, n_qubits: int):
    """Flip the phase of the single basis state |value> (n_qubits wide).

    Standard technique: X-gate every qubit whose target bit is 0, apply an
    (n_qubits-1)-controlled Z on the resulting all-ones pattern, then undo
    the X gates. This flips the phase of exactly the |value> basis state and
    nothing else.
    """
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    zero_positions = [i for i, b in enumerate(bits) if b == 0]

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


def build_oracle(marked_values, n_qubits):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for v in marked_values:
        apply_multi_controlled_z_for_value(qc, v, n_qubits)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(classical_marked, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

# Optimal number of Grover iterations, computed analytically from M and N.
r = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations r = {r}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(r):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on AerSimulator and check against the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical register bit order is little-endian in the returned
# bitstrings (c[N_QUBITS-1] ... c[0]); qubit i was measured into clbit i,
# and int(bitstring, 2) with the string as printed already reconstructs the
# integer n consistently with how apply_multi_controlled_z_for_value used
# qubit i as bit i, since Qiskit prints clbit (n_qubits-1) first.
marked_set = set(classical_marked)
prob_on_marked = 0.0
for bitstring, cnt in counts.items():
    n_value = int(bitstring, 2)
    if n_value in marked_set:
        prob_on_marked += cnt / shots

print(f"Total measured probability mass on classically-marked states: {prob_on_marked:.4f}")

# Also report the single most-sampled outcome as a sanity check.
best_bitstring = max(counts, key=counts.get)
best_n = int(best_bitstring, 2)
print(f"Most frequent measured n = {best_n} (gpf = {largest_prime_factor(best_n) if best_n else 'undefined'})")

PASS = prob_on_marked > 0.80 and largest_prime_factor(best_n) == TARGET_GPF

if PASS:
    print("PASS")
else:
    print("FAIL")
