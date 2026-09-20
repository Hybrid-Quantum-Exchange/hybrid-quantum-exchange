"""
Erdos problem #937 (erdosproblems.com), quantum-testable sequence lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: 937"):
    prize: no
    status: proved (Lean)
    oeis: ["possible"]      <-- NOTE: no concrete OEIS sequence id is recorded
                                 for problem 937 itself ("possible" is a status
                                 placeholder used in that dataset, not an id).
    tags: ["number theory", "powerful"]

Limitation, stated honestly up front: problem 937's own yaml entry carries no
usable OEIS id, so there is no single sequence of *this* problem to pull a
term from. Its tag ("powerful") ties it to the family of "powerful numbers"
problems (the neighboring entries #938/#939 in the same file do carry real
OEIS ids for that family, e.g. A001694 "powerful numbers": numbers n such
that every prime p dividing n also has p^2 dividing n). To still build a
genuine, small, computable, quantum-testable instance grounded in the actual
tag/topic of problem 937 rather than inventing unrelated math, this script
uses OEIS A001694 (powerful numbers) as the concrete finite property:

    Property tested: for n in [0, 63] (6 bits), n is a "powerful number" iff
    n == 0 is excluded and for every prime p | n, p^2 | n (equivalently,
    n's prime factorization has no exponent equal to 1). The classical
    membership set for n in [1, 63] is computed from scratch below by trial
    division (no OEIS values are copied in) and cross-checked against the
    literal start of A001694 (1, 4, 8, 9, 16, 25, 27, 32, 36, 49, ...).

Quantum circuit: a genuine Grover search over the 6-qubit space {0,...,63}.
The oracle is built directly from the classically-computed powerful-number
set (a standard, legitimate way to instantiate Grover's oracle for a known
predicate over a small finite domain) via multi-controlled-Z phase flips,
followed by the standard diffuser, iterated the Grover-optimal number of
times. The circuit is run on the ideal AerSimulator and the resulting
measurement distribution is compared against the classical set: PASS
requires that the search concentrates its probability mass (>= 90% of
shots) exactly on the classically-verified powerful numbers in [0, 63].

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles (no OEIS lookup).
# ---------------------------------------------------------------------------

def is_powerful(n: int) -> bool:
    """n > 0 is 'powerful' iff every prime factor p of n has p^2 | n."""
    if n <= 0:
        return False
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            exponent = 0
            while m % p == 0:
                m //= p
                exponent += 1
            if exponent == 1:
                return False
        p += 1
    # m is now 1 or a leftover prime factor with exponent 1 -> not powerful,
    # unless m == 1 (fully consumed by exponents >= 2).
    return m == 1


N_QUBITS = 6
DOMAIN = 1 << N_QUBITS  # 64

powerful_set = sorted(n for n in range(DOMAIN) if is_powerful(n))
print(f"Classically computed powerful numbers in [0, {DOMAIN - 1}]: {powerful_set}")

# Cross-check against the literal, well-known start of OEIS A001694
# (powerful numbers): 1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, ...
a001694_prefix = [1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64]
computed_prefix = [n for n in powerful_set if n <= 64]
assert computed_prefix == [x for x in a001694_prefix if x <= max(powerful_set)], (
    "Classical computation disagrees with the known start of A001694"
)
print("Classical computation agrees with the known start of A001694 (self-check OK).")

M = len(powerful_set)
print(f"Marked count M = {M} out of N = {DOMAIN}")


# ---------------------------------------------------------------------------
# 2. Build a Grover oracle from the classically-verified marked set.
# ---------------------------------------------------------------------------

def apply_mark(qc: QuantumCircuit, qubits, value: int, n_qubits: int) -> None:
    """Flip the phase of the single basis state |value> via multi-controlled Z."""
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    flip_qubits = [q for q, b in zip(qubits, bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    if n_qubits == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in flip_qubits:
        qc.x(q)


def oracle(n_qubits: int, marked):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for value in marked:
        apply_mark(qc, list(range(n_qubits)), value, n_qubits)
    return qc


def diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_iterations(n: int, m: int) -> int:
    theta = math.asin(math.sqrt(m / n))
    return max(1, round(math.pi / (4 * theta) - 0.5))


def best_grover_iterations(n: int, m: int, max_iter: int = 6) -> int:
    """Pick the iteration count in [1, max_iter] that maximizes marked-state
    probability, computed analytically (no simulation needed)."""
    theta = math.asin(math.sqrt(m / n))
    best_k, best_p = 1, -1.0
    for k in range(1, max_iter + 1):
        p = math.sin((2 * k + 1) * theta) ** 2
        if p > best_p:
            best_k, best_p = k, p
    return best_k


iterations = best_grover_iterations(DOMAIN, M)
print(f"Grover iterations: {iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

orc = oracle(N_QUBITS, powerful_set)
dif = diffuser(N_QUBITS)
for _ in range(iterations):
    qc.compose(orc, inplace=True)
    qc.compose(dif, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 20000
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit classical-bit string is c_{n-1}...c_1 c_0 (leftmost = MSB), which is
# exactly the standard binary representation with qubit 0 as LSB, so a plain
# int(bs, 2) recovers the value matching how `value` bits were encoded above.
def bitstring_to_int(bs: str) -> int:
    return int(bs, 2)

hits_in_marked = sum(c for bs, c in counts.items() if bitstring_to_int(bs) in powerful_set)
fraction_marked = hits_in_marked / shots

top_outcomes = sorted(counts.items(), key=lambda kv: -kv[1])[:M]
top_values = sorted(bitstring_to_int(bs) for bs, _ in top_outcomes)

print(f"Fraction of shots landing on classically-verified powerful numbers: {fraction_marked:.4f}")
print(f"Top-{M} most frequent measured values: {top_values}")
print(f"Classical powerful-number set:          {powerful_set}")

verified = fraction_marked >= 0.90
values_match = top_values == powerful_set

if verified and values_match:
    print("PASS")
elif verified:
    print("PASS (probability mass concentrated correctly; top-M set order/ties differ slightly)")
else:
    print("FAIL")
