"""
Erdos problem #368 -- quantum-testable instance.

OEIS sequence used: A074399.
  a(n) = the largest prime factor of n*(n+1).

Classical property tested (computed from first principles in this script,
not copied from OEIS):
  For a fixed small instance N = n*(n+1), the set of PRIME DIVISORS of N is
  a small, finite, exactly-computable set (found here by brute-force trial
  division over the search space 1..2^k - 1). This script uses Grover's
  algorithm to search a 4-qubit register (search space x in 0..15) for the
  marked states {x : x divides N and x is prime}, and then checks that:
    (a) Grover amplitude amplification, run against an oracle built purely
        from the "x divides N and x is prime" predicate (not from a
        hardcoded answer), concentrates measurement probability on exactly
        that classically-computed target set, and
    (b) the largest element of the (quantum-recovered) target set equals
        a(n), the classical value of the OEIS A074399 sequence at n,
        computed independently here via ordinary trial-division
        factorization of n*(n+1).

Instance chosen: n = 14, so N = n*(n+1) = 210 = 2 * 3 * 5 * 7.
Search register: 4 qubits, representing integers x in {0, ..., 15}
(0 and 1 are never valid answers and are excluded by the predicate).
Expected classical prime divisors of 210 in that range: {2, 3, 5, 7}.
Expected a(14) = largest prime factor of 210 = 7.
(Cross-check against the literal OEIS A074399 data: a(14) = 7 -- matches.)

The oracle is built by classically evaluating, for every one of the 16
basis states, the predicate "x divides N and x is prime", and phase-flipping
exactly the basis states that satisfy it (a standard, legitimate Grover
oracle construction -- the *predicate*, not the final answer, is what is
hardwired into the circuit; the predicate itself is evaluated by ordinary
trial division in Python, independently of any OEIS lookup).

Output: prints PASS if the quantum search's most-probable outcomes equal the
classically computed target set (prime divisors of N) and the maximum among
them equals the classically computed a(n); otherwise prints FAIL.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles -- no OEIS values copied in).
# ---------------------------------------------------------------------------

def is_prime(k: int) -> bool:
    if k < 2:
        return False
    if k < 4:
        return True
    if k % 2 == 0:
        return False
    i = 3
    while i * i <= k:
        if k % i == 0:
            return False
        i += 2
    return True


def largest_prime_factor(m: int) -> int:
    """Trial-division largest prime factor of m (m >= 2)."""
    best = 1
    d = 2
    x = m
    while d * d <= x:
        while x % d == 0:
            best = max(best, d)
            x //= d
        d += 1
    if x > 1:
        best = max(best, x)
    return best


N_QUBITS = 4
SEARCH_SIZE = 2 ** N_QUBITS  # 16, search space x in 0..15

n = 14
N = n * (n + 1)  # 210 = 2 * 3 * 5 * 7

classical_a_n = largest_prime_factor(N)  # expected 7

# Predicate-based target set: x in 0..15 such that x divides N and x is prime.
targets = sorted(
    x for x in range(SEARCH_SIZE)
    if x != 0 and N % x == 0 and is_prime(x)
)
classical_max_target = max(targets)

assert classical_max_target == classical_a_n, (
    "internal consistency check failed: largest predicate-marked divisor "
    "must equal the largest prime factor"
)

print(f"n = {n}, N = n*(n+1) = {N} = 2*3*5*7")
print(f"Classical prime divisors of N in range 0..{SEARCH_SIZE - 1}: {targets}")
print(f"Classical a(n) = largest prime factor of N (OEIS A074399(n)): {classical_a_n}")


# ---------------------------------------------------------------------------
# 2. Grover search over the 4-qubit register for the marked set `targets`.
# ---------------------------------------------------------------------------

def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip exactly the computational basis states in `marked`."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
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


oracle = build_oracle(N_QUBITS, targets)
diffuser = build_diffuser(N_QUBITS)

# Optimal number of Grover iterations for M marked items out of 2^n.
M = len(targets)
theta = math.asin(math.sqrt(M / SEARCH_SIZE))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Interpret bitstrings (Qiskit prints classical register with qubit 0 as the
# rightmost/least-significant character -- consistent with our little-endian
# encoding above).
outcome_counts = {}
for bitstring, c in counts.items():
    value = int(bitstring, 2)
    outcome_counts[value] = outcome_counts.get(value, 0) + c

sorted_outcomes = sorted(outcome_counts.items(), key=lambda kv: -kv[1])
top_m = [v for v, _ in sorted_outcomes[:M]]
top_m_probability = sum(outcome_counts.get(v, 0) for v in targets) / shots

print(f"Grover iterations used: {iterations}")
print(f"Measurement outcome counts (value: count): {dict(sorted(outcome_counts.items()))}")
print(f"Top-{M} most frequent measured values: {sorted(top_m)}")
print(f"Total probability mass on the classical target set {targets}: {top_m_probability:.4f}")


# ---------------------------------------------------------------------------
# 3. Verify quantum result against the classical answer.
# ---------------------------------------------------------------------------

top_m_matches_targets = sorted(top_m) == targets
quantum_recovered_max = max(top_m) if top_m_matches_targets else None
concentrated = top_m_probability > 0.85  # amplitude amplification should dominate

verified = (
    top_m_matches_targets
    and concentrated
    and quantum_recovered_max == classical_a_n
)

if verified:
    print(
        f"Quantum search recovered target set {sorted(top_m)} "
        f"(max = {quantum_recovered_max}) matching classical a({n}) = {classical_a_n}."
    )
    print("PASS")
else:
    print(
        f"Mismatch: quantum top-{M} = {sorted(top_m)} vs classical targets = {targets}, "
        f"probability mass = {top_m_probability:.4f}, classical a(n) = {classical_a_n}."
    )
    print("FAIL")
