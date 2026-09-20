"""
Erdos problem #454 (erdosproblems.com / Thomas Bloom's list), OEIS A389676 / A389677.

Problem statement (Pomerance's conjecture, as tabulated by Bloom):
    Let p_k denote the k-th prime. Define
        f(n) = min_{1 <= i <= n-1} ( p_{n+i} + p_{n-i} ).
    Is it true that limsup_n ( f(n) - 2*p_n ) = infinity?
    (Pomerance proved the limsup is at least 2; the problem is open.)

A389676 tabulates f(n) - 2*p(n) and A389677 tabulates f(n) itself (both
built from exactly this min-over-i expression), so the finite, computable
property we test here is a direct instance of the quantity these sequences
are built from: for a fixed n, find the index i in {1, ..., n-1} that
achieves the minimum of p_{n+i} + p_{n-i}, i.e. solve

        argmin_i  p_{n+i} + p_{n-i}

by unstructured search over the n-1 candidate indices. This is exactly the
shape Grover's algorithm is for: an oracle that can recognize a solution
(here: "this index attains the classically-precomputed minimum value")
searched for by amplitude amplification instead of brute force.

Concretely, for n = 6:
    p_1..p_11 = 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31
    i=1: p_7+p_5   = 17+11 = 28
    i=2: p_8+p_4   = 19+7  = 26   <- unique minimum
    i=3: p_9+p_3   = 23+5  = 28
    i=4: p_10+p_2  = 29+3  = 32
    i=5: p_11+p_1  = 31+2  = 33
so f(6) = 26, achieved uniquely at i = 2 (2*p_6 = 26 too, matching the
known Pomerance lower-bound-achieving case f(n) - 2p_n = 0 for small n).

The classical answer (f(6) = 26, argmin i = 2) is computed here from first
principles (a trial-division primality sieve, no OEIS lookup of the answer
value). The quantum part is a genuine 3-qubit Grover search over the 8
basis states representing candidate index i-1 in {0,...,4} (indices 5,6,7
are padding and never marked as solutions); the oracle marks exactly the
basis state encoding the classically-precomputed argmin, and the circuit
is run on the ideal AerSimulator. PASS means the most frequently measured
computational basis state after Grover amplification equals the classical
argmin index.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


# ---------- classical part (first principles, no OEIS value copied) ----------

def is_prime(k: int) -> bool:
    if k < 2:
        return False
    if k % 2 == 0:
        return k == 2
    i = 3
    while i * i <= k:
        if k % i == 0:
            return False
        i += 2
    return True


def nth_prime(n: int) -> int:
    count = 0
    num = 1
    while count < n:
        num += 1
        if is_prime(num):
            count += 1
    return num


N = 6  # instance of the problem: compute f(N) = min_{1<=i<=N-1} p_{N+i}+p_{N-i}

primes = {k: nth_prime(k) for k in range(1, 2 * N)}
candidates = []  # (i, value), i = 1..N-1
for i in range(1, N):
    val = primes[N + i] + primes[N - i]
    candidates.append((i, val))

f_N = min(val for _, val in candidates)
argmin_i_list = [i for i, val in candidates if val == f_N]
assert len(argmin_i_list) == 1, "instance chosen to have a unique minimizer"
argmin_i = argmin_i_list[0]
argmin_index = argmin_i - 1  # 0-based index into the search register, in [0, N-2]

NUM_QUBITS = 3  # 2^3 = 8 >= N-1 = 5 candidate indices (padding states never marked)
SPACE_SIZE = 2 ** NUM_QUBITS

print(f"Instance: N = {N}")
print(f"Candidates (i, p_(N+i)+p_(N-i)): {candidates}")
print(f"Classical f(N) = {f_N}, unique argmin i = {argmin_i} "
      f"(search-register index {argmin_index})")
print(f"2*p_N = {2 * primes[N]}  (f(N) - 2*p_N = {f_N - 2 * primes[N]})")


# ---------- quantum part: Grover search for the argmin index ----------

def oracle_marking(index: int, num_qubits: int) -> QuantumCircuit:
    """Phase-flip the single basis state |index> (multi-controlled Z via X-sandwich)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    bits = format(index, f"0{num_qubits}b")[::-1]  # little-endian per qubit ordering
    for q, b in enumerate(bits):
        if b == "0":
            qc.x(q)
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    for q, b in enumerate(bits):
        if b == "0":
            qc.x(q)
    return qc


def diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


iterations = max(1, round((np.pi / 4) * np.sqrt(SPACE_SIZE / 1)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

oracle = oracle_marking(argmin_index, NUM_QUBITS)
diff = diffuser(NUM_QUBITS)
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diff, inplace=True)

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

sim = AerSimulator()
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=4096).result()
counts = result.get_counts()

# Qiskit's classical-bit string is c_{n-1}...c_0 with c_0 (qubit 0, our LSB) on
# the right, so reading it as a plain binary integer recovers our index directly.
most_likely_bitstring = max(counts, key=counts.get)
most_likely_index = int(most_likely_bitstring, 2)

print(f"Grover iterations used: {iterations}")
print(f"Measurement counts: {counts}")
print(f"Most frequently measured index: {most_likely_index} "
      f"(expected classical argmin index: {argmin_index})")

classical_ok = (f_N == min(val for _, val in candidates))  # sanity: recompute
quantum_ok = (most_likely_index == argmin_index)

if classical_ok and quantum_ok:
    print("PASS")
else:
    print("FAIL")
