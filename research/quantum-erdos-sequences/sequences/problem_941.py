"""
Erdos problem #941 -- quantum-testable instance.

Source metadata (erdosproblems.com data, problem 941):
    prize: no
    status: proved (2025-08-31)
    oeis: ["A056828"]
    tags: ["number theory", "powerful"]

OEIS id used: A056828.
A056828 is the sequence of POWERFUL NUMBERS (squarefull numbers): positive
integers n such that for every prime p dividing n, p^2 also divides n.
Equivalently, n is powerful iff n can be written as a^2 * b^3 for some
nonnegative integers a, b >= 1 (with b squarefree). The first terms are
1, 4, 8, 9, 16, 25, 27, 32, 36, 49, ...

Classical property tested here (small, finite, computable):
    For n in {0, 1, ..., 15} (4 bits), is n a powerful number, where
    "powerful" means: n >= 1 AND for every prime p | n, p^2 | n.
    (n = 0 is treated as not powerful / out of scope; it never occurs as a
    valid factorization target.)

The classical answer is computed from first principles in this script
(trial-division factorization, no OEIS lookup, no external data) for every
n in the search space, giving the exact marked set:
    powerful numbers in [0, 15] = {1, 4, 8, 9}

Quantum approach: Grover's search.
We build a 4-qubit Grover search whose oracle phase-flips exactly the
computational basis states |n> for n in the classically-computed marked
set {1, 4, 8, 9} (implemented as a diagonal phase oracle, built directly
from the classical predicate -- no shortcut, no hard-coded quantum "answer"
beyond the oracle construction itself). We run the optimal number of Grover
iterations for this database size (N=16, M=4 marked -> 1 iteration) on the
ideal AerSimulator and check that the highest-probability measured
outcomes exactly reproduce the classically-known set of powerful numbers
in [0, 15].

PASS criterion: the set of the top-M most frequent measurement outcomes
(M = number of marked items) returned by the quantum circuit equals the
classically computed set of powerful numbers in [0, 15].
"""

from __future__ import annotations

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup)
# ---------------------------------------------------------------------------

def is_powerful(n: int) -> bool:
    """n >= 1 is powerful iff every prime factor of n divides n at least twice."""
    if n < 1:
        return False
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            count = 0
            while m % p == 0:
                m //= p
                count += 1
            if count < 2:
                return False
        p += 1
    if m > 1:
        # m is a prime factor left over; it must appear at least squared in n.
        # Since we've already divided out all smaller factors, m appears to
        # the first power in what's left, so check its total exponent in n.
        exp = 0
        t = n
        while t % m == 0:
            t //= m
            exp += 1
        if exp < 2:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # search space {0, ..., 15}

classical_powerful = sorted(n for n in range(N) if is_powerful(n))
print(f"Classical search space: n in [0, {N - 1}]")
print(f"Classically computed powerful numbers (A056828 members here): {classical_powerful}")

MARKED = set(classical_powerful)
M = len(MARKED)
assert M > 0, "no marked items found -- cannot build a nontrivial Grover search"


# ---------------------------------------------------------------------------
# 2. Grover oracle: diagonal phase flip on exactly the marked basis states
# ---------------------------------------------------------------------------

def build_oracle(n_qubits: int, marked: set[int]) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for target in marked:
        bits = format(target, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
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
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# 3. Assemble and run the full Grover circuit
# ---------------------------------------------------------------------------

oracle = build_oracle(N_QUBITS, MARKED)
diffuser = build_diffuser(N_QUBITS)

# Optimal number of Grover iterations for database size N, M marked items.
iterations = max(1, round((math.pi / 4) * math.sqrt(N / M) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))

print(f"\nGrover circuit: {N_QUBITS} qubits, {M} marked items, {iterations} iteration(s)")

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit prints the classical register as a string with the leftmost
# character being the highest-index classical bit (c[n-1] ... c[0]). Our
# measure(range, range) maps qubit i -> clbit i, so this string, read
# directly as a binary number, already equals the integer n with qubit 0
# as the least-significant bit.
def bitstring_to_int(bs: str) -> int:
    return int(bs, 2)

value_counts: Counter[int] = Counter()
for bitstring, cnt in counts.items():
    value_counts[bitstring_to_int(bitstring)] += cnt

top_values = {v for v, _ in value_counts.most_common(M)}

print(f"Top-{M} most frequent measured values: {sorted(top_values)}")
print(f"Full measurement histogram (value: count): {dict(sorted(value_counts.items()))}")

# ---------------------------------------------------------------------------
# 4. Compare quantum result to classical answer
# ---------------------------------------------------------------------------

quantum_matches_classical = top_values == MARKED

print(f"\nClassical answer : {sorted(MARKED)}")
print(f"Quantum answer   : {sorted(top_values)}")

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
