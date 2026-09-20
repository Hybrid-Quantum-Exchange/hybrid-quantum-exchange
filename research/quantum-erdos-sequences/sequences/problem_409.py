"""
Erdos problem #409 -- quantum-testable instance.

Problem #409 (erdosproblems.com) asks about the iteration n -> phi(n)+1
(Euler's totient plus one): how many steps F(n) are required, starting from
n, to first reach a prime; whether infinitely many n converge to the same
prime; and what density of integers converge to a given prime.

OEIS ids used:
  - A039651: a(n) = number of iterations of f(x) = phi(x)+1 on n required to
    reach a prime. First terms (n=1..15):
    1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 2
    (a(15) = 2 is the term this script targets and verifies.)
  - A229487: conjectured greatest number that converges to prime(n) under
    the same iteration (not used quantitatively here beyond context).

Classical property tested
--------------------------
Over the small search space n in {0, 1, ..., 15} (encodable in 4 qubits),
define a(n) = number of iterations of f(x) = phi(x)+1, starting at x = n,
required until the value is prime (a(n) is left undefined/treated as "not
matching" for n = 0 and n = 1, since phi is not meaningfully iterated from
0 and 1 already needs no iteration in the OEIS b-file convention used
above -- we simply compute f directly from first principles and iterate
until a prime appears, capping the iteration count so the search stays
finite).

This script computes, from first principles (its own Euler's totient
function and its own primality test -- no OEIS values are copied), which
n in {0, ..., 15} satisfy a(n) == 2. Cross-checking against the published
A039651 terms above confirms the unique answer is n = 15
(15 -> phi(15)+1 = 9 -> phi(9)+1 = 7, and 7 is prime, in exactly 2 steps;
every other n in range reaches a prime in 0, 1, or fails within the cap).

Quantum circuit
----------------
A genuine Grover search over the 4-qubit space {0,...,15} is built. The
oracle phase-flips exactly the marked basis state(s) determined classically
above (the n with a(n) == 2), and the diffusion (inversion-about-mean)
operator amplifies it. The circuit is run on the ideal AerSimulator, and
the most frequently measured 4-bit string is compared against the
classically computed marked state. The script prints PASS if the quantum
search recovers the correct classical answer, FAIL otherwise.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical number theory, from first principles.
# ---------------------------------------------------------------------

def euler_phi(x: int) -> int:
    """Euler's totient function, computed from first principles."""
    if x <= 0:
        raise ValueError("phi undefined for x <= 0")
    if x == 1:
        return 1
    result = x
    n = x
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result


def is_prime(x: int) -> bool:
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0:
        return False
    p = 3
    while p * p <= x:
        if x % p == 0:
            return False
        p += 2
    return True


def iterations_to_prime(n: int, cap: int = 20) -> int:
    """
    Number of iterations of f(x) = phi(x)+1 starting at n required to reach
    a prime, matching the definition behind OEIS A039651. Returns -1 if no
    prime is reached within `cap` iterations (not expected for n<=15) or if
    n is too small for phi to be meaningfully iterated (n <= 1).
    """
    if n <= 1:
        return -1
    x = n
    if is_prime(x):
        return 0
    for steps in range(1, cap + 1):
        x = euler_phi(x) + 1
        if is_prime(x):
            return steps
    return -1


# ---------------------------------------------------------------------
# 2. Classical search: find n in {0,...,15} with iterations_to_prime(n) == 2.
# ---------------------------------------------------------------------

N_QUBITS = 4
SEARCH_SPACE = list(range(2 ** N_QUBITS))  # 0..15

TARGET_ITERATIONS = 2

marked = [n for n in SEARCH_SPACE if iterations_to_prime(n) == TARGET_ITERATIONS]

print(f"Classical scan over n = 0..15 for a(n) == {TARGET_ITERATIONS} "
      f"under f(x) = phi(x)+1:")
for n in SEARCH_SPACE:
    print(f"  n={n:2d}  a(n)={iterations_to_prime(n)}")
print(f"Marked (classical) states: {marked}")

if len(marked) != 1:
    raise RuntimeError(
        f"Expected a unique marked state in this small instance, got {marked}. "
        "Refusing to build a Grover oracle for a non-unique target."
    )

classical_answer = marked[0]
print(f"Classical answer: n = {classical_answer} "
      f"(binary {classical_answer:0{N_QUBITS}b})")


# ---------------------------------------------------------------------
# 3. Quantum circuit: Grover search over the 4-qubit space for the marked n.
# ---------------------------------------------------------------------

def build_oracle(marked_value: int, n_qubits: int) -> QuantumCircuit:
    """Phase-flip the single basis state |marked_value>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked_value, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
    # Flip qubits that should be 0 in the marked state, so the marked
    # state maps to |11...1>, apply a multi-controlled Z, then flip back.
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


n_items = 2 ** N_QUBITS
n_marked = len(marked)
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(n_items / n_marked)))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(classical_answer, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

for _ in range(optimal_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings as c[n-1]...c[0]; our oracle indexed qubit i as
# the i-th bit of the value (little-endian), so reverse before parsing.
tallies = Counter()
for bitstring, count in counts.items():
    value = int(bitstring[::-1], 2)
    tallies[value] += count

most_common_value, most_common_count = tallies.most_common(1)[0]

print(f"\nGrover search used {optimal_iterations} iteration(s) over "
      f"{n_items} basis states.")
print(f"Quantum measurement distribution (top 5): {tallies.most_common(5)}")
print(f"Most frequently measured value: {most_common_value} "
      f"({most_common_count}/{shots} shots)")

verified = (most_common_value == classical_answer)

print(f"\nClassical answer : n = {classical_answer}")
print(f"Quantum answer   : n = {most_common_value}")
print("PASS" if verified else "FAIL")
