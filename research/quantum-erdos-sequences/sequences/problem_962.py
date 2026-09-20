"""
Erdos problem #962 (erdosproblems.com), quantum-testable sequence lane.

OEIS id used: A327909
  "a(n) is the smallest start of a run of n or more consecutive integers
  each having a prime factor greater than n."
  (equivalently: none of the n consecutive integers is 1 or a power of 2,
  since "no prime factor greater than n" for n=2 means the number's only
  prime factor is <= 2, i.e. it is a power of 2, or it is 1 which has no
  prime factors at all)

Classical property tested here (computed from first principles, not copied
from OEIS): for n = 2, find the smallest positive integer k such that both
k and k+1 have a prime factor strictly greater than 2 (i.e. neither k nor
k+1 is 1 or a power of two). OEIS lists a(2) = 5. We recompute this
classically in this script by brute-force factorization, then encode the
same predicate as a quantum oracle over a small search space and use
Grover's algorithm to locate a marked (satisfying) k, and check the
smallest such k matches the classically-derived a(2).

Search space: k in {0, 1, ..., 7}, represented by 3 qubits (k=0 is a
placeholder/out-of-range value, never marked, since the sequence indexes
from k>=1).

Grover oracle: phase-flips computational basis states |k> for which
is_power_of_two_or_one(k) is False AND is_power_of_two_or_one(k+1) is
False (both computed classically to build the fixed marked set, then
hard-coded into a multi-controlled-Z oracle over the 3 index qubits).

This is a genuine (if small) instance of amplitude amplification: Grover's
algorithm boosts the amplitude of the marked basis states {5, 6} out of an
unstructured superposition over 8 states, which is exactly the "small
search space whose answer is a known term" shape called for -- a(2) = 5
is the smallest marked state.

Run: python3 problem_962.py
Prints PASS if the Grover circuit's most-sampled outcome is a marked state
and the smallest marked state (found purely classically) equals the
classically-recomputed a(2) = 5.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate
import numpy as np


def prime_factors(m: int):
    """Return the set of prime factors of m (m >= 2), by trial division."""
    factors = set()
    x = m
    d = 2
    while d * d <= x:
        while x % d == 0:
            factors.add(d)
            x //= d
        d += 1
    if x > 1:
        factors.add(x)
    return factors


def max_prime_factor(m: int):
    """Largest prime factor of m, or None if m has no prime factors (m==1)."""
    if m <= 1:
        return None
    return max(prime_factors(m))


def has_prime_factor_greater_than(m: int, n: int) -> bool:
    mpf = max_prime_factor(m)
    return mpf is not None and mpf > n


def classical_a(n: int, search_limit: int = 200) -> int:
    """
    Classical, from-first-principles computation of A327909(n): smallest k
    such that k, k+1, ..., k+n-1 all have a prime factor > n.
    """
    k = 1
    while k <= search_limit:
        if all(has_prime_factor_greater_than(k + j, n) for j in range(n)):
            return k
        k += 1
    raise RuntimeError(f"no a({n}) found within search_limit={search_limit}")


# ---------------------------------------------------------------------------
# Step 1: classical ground truth for n = 2, on a small finite instance.
# ---------------------------------------------------------------------------
N = 2
K_MAX = 7  # search space: k in 0..7 (3 qubits)

a_n_classical = classical_a(N, search_limit=K_MAX)
print(f"Classical A327909({N}) computed from first principles = {a_n_classical}")

# Build the marked set over the finite index space {0,...,K_MAX} the same
# predicate Grover will amplify: k such that k and k+1 both have a prime
# factor > N. (k=0 is never marked: 0 has no well-defined prime factors.)
marked = []
for k in range(K_MAX + 1):
    if k >= 1 and all(has_prime_factor_greater_than(k + j, N) for j in range(N)):
        if k + N - 1 <= K_MAX + 1:  # stay in-range for the check above
            marked.append(k)

print(f"Marked k in [0,{K_MAX}] satisfying the A327909(N={N}) predicate: {marked}")
assert marked, "expected at least one marked state"
assert min(marked) == a_n_classical, "smallest marked state must equal a(n)"

# ---------------------------------------------------------------------------
# Step 2: Grover search over 3 qubits (k = 0..7) for the marked predicate.
# ---------------------------------------------------------------------------
NUM_QUBITS = 3
assert K_MAX < 2 ** NUM_QUBITS


def oracle_circuit(marked_states, num_qubits):
    qc = QuantumCircuit(num_qubits, name="oracle")
    mcz = MCMTGate(ZGate(), num_qubits - 1, 1)
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.append(mcz, list(range(num_qubits)))
        for i in zero_positions:
            qc.x(i)
    return qc


def diffuser_circuit(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    mcz = MCMTGate(ZGate(), num_qubits - 1, 1)
    qc.append(mcz, list(range(num_qubits)))
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


oracle = oracle_circuit(marked, NUM_QUBITS)
diffuser = diffuser_circuit(NUM_QUBITS)

# Optimal number of Grover iterations for M marked out of N=2^n states.
N_states = 2 ** NUM_QUBITS
M = len(marked)
theta = np.arcsin(np.sqrt(M / N_states))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations} (N={N_states} states, M={M} marked)")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUMBERS := NUM_QUBITS))

sim = AerSimulator()
tqc = transpile(qc, sim)
job = sim.run(tqc, shots=4096)
result = job.result()
counts = result.get_counts()

# Convert bitstrings (Qiskit little-endian: rightmost char is qubit 0) to ints.
int_counts = {}
for bitstring, c in counts.items():
    k_val = int(bitstring, 2)
    int_counts[k_val] = int_counts.get(k_val, 0) + c

print("Measurement outcomes (k -> count):", dict(sorted(int_counts.items())))

most_likely_k = max(int_counts, key=int_counts.get)
print(f"Most frequently measured k = {most_likely_k}")

# ---------------------------------------------------------------------------
# Step 3: verify quantum result against the classical answer.
# ---------------------------------------------------------------------------
quantum_found_marked = most_likely_k in marked
smallest_marked_matches_a_n = (min(marked) == a_n_classical)

verified = quantum_found_marked and smallest_marked_matches_a_n

print(f"Quantum-found state is marked: {quantum_found_marked}")
print(f"min(marked) == classical a({N}) == {a_n_classical}: {smallest_marked_matches_a_n}")

if verified:
    print("PASS")
else:
    print("FAIL")
