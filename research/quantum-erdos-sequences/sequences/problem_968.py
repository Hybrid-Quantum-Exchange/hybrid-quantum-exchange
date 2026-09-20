"""
Erdos problem #968 -- quantum-testable instance
=================================================

Erdos problem #968 (data/problems.yaml, tags: ["number theory"]) is linked to
OEIS sequence A387591: primes prime(k) such that

    (k+1) * prime(k)  <  k * prime(k+1)

i.e. the k-th prime for which the ratio prime(k)/k is strictly less than
prime(k+1)/(k+1) (the "prime(k)/k" ratio is locally increasing at index k).
The sequence starts 3, 5, 7, 13, 19, 23, 31, 37, 43, 47, 53, ...
(prime(2), prime(3), prime(4), prime(6), prime(8), prime(9), prime(11), ...).

Classical property tested here
-------------------------------
For k = 1..8 (the first N = 8 primes, 3 qubits worth of index space), decide
membership: does index k satisfy (k+1)*prime(k) < k*prime(k+1)?  This is
computed from first principles below with a plain trial-division prime
generator -- no OEIS values are copied, only derived and then cross-checked
against the sequence's published terms above.

Over k=1..8 the inequality happens to hold for a majority of indices
({2,3,4,6,8}, i.e. prime(k) in {3,5,7,13,19} -- matching A387591's own
published prefix). Standard Grover search amplifies a *minority* marked set,
so the quantum circuit below searches for the complementary minority
property instead: k in {1,...,8} for which the inequality FAILS, i.e.
(k+1)*prime(k) >= k*prime(k+1) (this happens for k in {1,5,7}). This is the
same underlying arithmetic condition, just targeting its negation so that
the search space is minority-marked -- a legitimate, finite, first-principles
property directly derived from the sequence's defining inequality.

Quantum approach
-----------------
A Grover search circuit over the 3-qubit index register {0,1}^3 (representing
k-1 for k=1..8). An oracle, built directly from the classically-computed
marked set (no external oracle library), flags exactly the indices satisfying
the inequality by phase-flipping those computational basis states. One Grover
iteration (oracle + diffusion) amplifies the marked amplitudes. We then check
on the ideal AerSimulator that:
  1. the total measured probability mass on the marked indices exceeds the
     uniform-superposition baseline (Grover amplification worked), and
  2. the set of the M most frequently measured outcomes (M = number of marked
     indices) equals exactly the classically-computed marked set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup)
# ---------------------------------------------------------------------------

def first_n_primes(n: int) -> list[int]:
    """Trial-division prime generator; returns the first n primes."""
    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


N = 8  # search space size: k = 1..N  -> needs ceil(log2(N)) = 3 qubits
NUM_QUBITS = int(np.ceil(np.log2(N)))
assert 2 ** NUM_QUBITS == N, "N must be a power of two for this simple index register"

# need primes up to index N+1 to evaluate the inequality at k = N
primes = first_n_primes(N + 1)  # primes[0] = prime(1), ..., primes[N] = prime(N+1)


def satisfies(k: int) -> bool:
    """(k+1)*prime(k) < k*prime(k+1), 1-indexed."""
    pk = primes[k - 1]
    pk1 = primes[k]
    return (k + 1) * pk < k * pk1


classical_hit_k = [k for k in range(1, N + 1) if satisfies(k)]
classical_hit_terms = [primes[k - 1] for k in classical_hit_k]

# Grover search performs best (and the standard diffuser amplifies rather
# than suppresses) when the marked fraction M/N < 1/2. For k = 1..N here the
# inequality holds for a *majority* of indices, so we instead search for the
# complementary, minority property: k for which the inequality FAILS, i.e.
# (k+1)*prime(k) >= k*prime(k+1) -- equally a genuine, finite, first-principles
# property of the same A387591-defining condition (its negation restricted to
# this finite range).
classical_marked_k = [k for k in range(1, N + 1) if not satisfies(k)]
classical_marked_terms = [primes[k - 1] for k in classical_marked_k]
marked_indices = [k - 1 for k in classical_marked_k]  # 0-indexed register values

print(f"Primes used: {primes}")
print(f"k (1..{N}) satisfying the A387591 inequality (in-sequence terms): {classical_hit_k}")
print(f"Corresponding A387591 terms: {classical_hit_terms}")
print(f"k (1..{N}) FAILING the inequality (Grover search target, minority set): {classical_marked_k}")
print(f"Marked register indices (0-indexed): {marked_indices}")

# Sanity cross-check against the sequence's published start (3,5,7,13,...):
# the first six terms of A387591 are prime(2),prime(3),prime(4),prime(6),
# prime(8),prime(9) = 3,5,7,13,19,23 -- confirm our k=1..8 computation of the
# *in-sequence* set agrees with those that fall in range.
expected_prefix = [3, 5, 7, 13, 19, 23]
common_len = min(len(expected_prefix), len(classical_hit_terms))
assert classical_hit_terms[:common_len] == expected_prefix[:common_len], (
    "classical computation disagrees with A387591's published terms"
)


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the marked index set
# ---------------------------------------------------------------------------

def build_oracle(num_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each index in `marked` (multi-controlled Z)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for idx in marked:
        bits = format(idx, f"0{num_qubits}b")
        # flip qubits that should be 0 for this basis state, so an all-ones
        # control pattern corresponds to |idx>
        zero_positions = [i for i, b in enumerate(bits[::-1]) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


M = len(marked_indices)
theta = np.arcsin(np.sqrt(M / N))
# optimal number of Grover iterations
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

oracle = build_oracle(NUM_QUBITS, marked_indices)
diffuser = build_diffuser(NUM_QUBITS)
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 8192
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer
# ---------------------------------------------------------------------------

# probability mass landing on marked vs. unmarked indices
marked_bitstrings = {format(i, f"0{NUM_QUBITS}b") for i in marked_indices}
marked_prob = sum(c for b, c in counts.items() if b in marked_bitstrings) / shots
uniform_baseline = M / N

# the M most frequent measured outcomes
top_m = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:M]
top_m_indices = sorted(int(b, 2) for b, _ in top_m)

print(f"Grover iterations used: {iterations}")
print(f"Counts: {counts}")
print(f"Probability mass on marked states: {marked_prob:.4f} (uniform baseline {uniform_baseline:.4f})")
print(f"Top-{M} measured indices: {top_m_indices}")
print(f"Classical marked indices: {sorted(marked_indices)}")

amplified = marked_prob > uniform_baseline
matches_classical = top_m_indices == sorted(marked_indices)

if amplified and matches_classical:
    print("PASS")
else:
    print("FAIL")
