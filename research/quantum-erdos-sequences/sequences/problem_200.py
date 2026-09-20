"""
Erdos problem #200 (https://www.erdosproblems.com/200)
OEIS id used: A005115.

A005115(n) = the smallest prime that is the LAST term of an n-term
arithmetic progression of primes, minimized over all such progressions
(i.e. over start value a >= 1 and common difference d >= 1, take the
progression a, a+d, a+2d, ..., a+(n-1)d in which every term is prime and
whose final term a+(n-1)d is as small as possible).

Classical property tested here (n = 4 terms):
    Over the small finite search space a in {1,...,7}, d in {1,...,7},
    find the pair (a, d) such that a, a+d, a+2d, a+3d are ALL prime.
    A005115(4) = 23, realized uniquely (within this search space) by
    a = 5, d = 6: the progression 5, 11, 17, 23 (all four terms prime),
    whose last term is 23 -- matching the known OEIS value A005115(4)=23.

This script:
  1. Computes classically, from first principles (trial division, no
     OEIS lookup), which (a, d) pairs in the 7x7 grid give four primes
     in arithmetic progression, and confirms there is exactly one such
     pair, with last term 23 (matching A005115(4)).
  2. Encodes a in 3 qubits and d in 3 qubits (values 1..7 stored as the
     3-bit binary value itself, 0 unused/excluded by construction since
     the search only ever marks a,d in 1..7 and 0 never satisfies the
     all-prime condition because 0 is not prime), builds a Grover oracle
     that phase-flips exactly the basis state(s) satisfying the
     classical primality condition (the marking itself is derived from
     the classical primality check, not hard-coded from OEIS), and runs
     Grover's algorithm with the optimal number of iterations for a
     single marked state out of 64.
  3. Measures the resulting distribution on the ideal AerSimulator and
     checks that the most frequent outcome decodes to (a=5, d=6), i.e.
     the progression ending at 23 -- printing PASS/FAIL accordingly.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS values copied in).
# ---------------------------------------------------------------------------
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in range(2, int(n**0.5) + 1):
        if n % p == 0:
            return False
    return True


A_RANGE = range(1, 8)  # 3 bits: 1..7 (0 excluded, never a solution)
D_RANGE = range(1, 8)

classical_solutions = []
for a in A_RANGE:
    for d in D_RANGE:
        terms = [a + i * d for i in range(4)]
        if all(is_prime(t) for t in terms):
            classical_solutions.append((a, d, terms))

assert len(classical_solutions) == 1, (
    f"Expected exactly one solution in the 7x7 grid, found {classical_solutions}"
)
CLASSICAL_A, CLASSICAL_D, CLASSICAL_TERMS = classical_solutions[0]
CLASSICAL_LAST_TERM = CLASSICAL_TERMS[-1]

print("Classical search over a,d in 1..7:")
print(f"  solution(s): {classical_solutions}")
print(f"  A005115(4) candidate (last term of the unique AP found): {CLASSICAL_LAST_TERM}")
assert CLASSICAL_LAST_TERM == 23, "This should reproduce OEIS A005115(4) = 23"


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle marking the unique (a, d) satisfying the
#    property above. a and d are each stored in 3 qubits as their plain
#    3-bit binary representation (values 0..7); the oracle is derived
#    programmatically from CLASSICAL_A / CLASSICAL_D, not hand-picked.
# ---------------------------------------------------------------------------
N_BITS_A = 3
N_BITS_D = 3
N_QUBITS = N_BITS_A + N_BITS_D  # 6 qubits, 64 basis states total

# qubit layout: [a2 a1 a0 d2 d1 d0]  (a occupies qubits 0-2, d qubits 3-5)


def marked_bitstring(a: int, d: int) -> str:
    """Little-endian bit layout matching the qubit indices used below."""
    a_bits = format(a, "03b")[::-1]  # a0 a1 a2 order -> qubit0..qubit2
    d_bits = format(d, "03b")[::-1]
    return a_bits + d_bits  # index 0..2 = a bits, index 3..5 = d bits


MARKED_BITS = marked_bitstring(CLASSICAL_A, CLASSICAL_D)


def build_oracle(bits: str) -> QuantumCircuit:
    """Phase-flip the single computational basis state 'bits' (multi-controlled Z)."""
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    for i in zero_positions:
        qc.x(i)
    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(MARKED_BITS)
diffuser = build_diffuser(N_QUBITS)

# Optimal number of Grover iterations for 1 marked state out of 2^N_QUBITS.
N_STATES = 2 ** N_QUBITS
n_iterations = max(1, round(math.pi / 4 * math.sqrt(N_STATES / 1)))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(n_iterations):
    qc.append(oracle.to_instruction(), range(N_QUBITS))
    qc.append(diffuser.to_instruction(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and check the result.
# ---------------------------------------------------------------------------
sim = AerSimulator()
compiled = transpile(qc, sim)
result = sim.run(compiled, shots=2048).result()
counts = result.get_counts()

# Qiskit's classical-register bitstring is printed MSB-first (qubit N-1 ... qubit 0).
top_outcome = max(counts, key=counts.get)
top_outcome_little_endian = top_outcome[::-1]  # index 0..5 matches qubit index

decoded_a_bits = top_outcome_little_endian[0:3][::-1]
decoded_d_bits = top_outcome_little_endian[3:6][::-1]
decoded_a = int(decoded_a_bits, 2)
decoded_d = int(decoded_d_bits, 2)
decoded_last_term = decoded_a + 3 * decoded_d

print(f"\nGrover iterations used: {n_iterations}")
print(f"Most frequent measured outcome: {top_outcome} "
      f"(count {counts[top_outcome]}/{sum(counts.values())})")
print(f"Decoded (a, d) = ({decoded_a}, {decoded_d}), last term = {decoded_last_term}")
print(f"Classical target (a, d) = ({CLASSICAL_A}, {CLASSICAL_D}), last term = {CLASSICAL_LAST_TERM}")

verified = (
    decoded_a == CLASSICAL_A
    and decoded_d == CLASSICAL_D
    and decoded_last_term == CLASSICAL_LAST_TERM == 23
)

if verified:
    print("\nPASS: Grover search found the unique 4-term prime arithmetic progression "
          "(5, 11, 17, 23), matching OEIS A005115(4) = 23.")
else:
    print("\nFAIL: quantum result did not match the classical property.")
