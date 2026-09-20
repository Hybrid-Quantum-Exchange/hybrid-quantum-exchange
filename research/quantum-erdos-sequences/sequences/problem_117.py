"""
Erdos problem #117 -- quantum-testable sequence entry.

HONESTY NOTE ON SCOPE
----------------------
In data/problems.yaml (manman4/erdosproblems, read-only clone), problem 117 is
recorded as:

    number: "117"
    tags: ["group theory"]
    oeis: ["possible"]
    informal_status: open (as of 2025-08-31)

"possible" is a placeholder value used by that dataset when an OEIS link has
not actually been established -- it is not a real OEIS sequence id, and the
dataset carries no problem statement text for #117 (only status/tag
metadata). So there is no genuine OEIS id to derive a property from for this
specific problem, and no way to honestly "verify against" a nonexistent
sequence.

Rather than fabricate a link to problem 117 that doesn't exist, this script
uses the one real piece of information available -- the tag "group theory"
-- to build a genuine, classically-checkable finite decision problem from
elementary group theory, and a real Grover search circuit that finds it. The
mathematical content below is real and independently verifiable; what is
NOT claimed is that it is *the* sequence behind Erdos problem 117.

THE CLASSICAL PROPERTY (real, well-known theorem)
---------------------------------------------------
A positive integer n is called a "cyclic number" if every group of order n
is cyclic. A classical theorem (see OEIS A003277, "Cyclic numbers: n such
that gcd(n, phi(n)) = 1") states:

    n is a cyclic number  <=>  gcd(n, phi(n)) = 1

where phi is Euler's totient function.

Small instance chosen here: search space n in {1, 2, ..., 16} (4 qubits,
index i encodes n = i + 1). Target property: "n is odd AND n is NOT a
cyclic number" (i.e. gcd(n, phi(n)) != 1), i.e. n is an odd order for which
a non-cyclic (necessarily non-abelian, since odd order forces non-cyclic to
be non-abelian by a classical fact) group actually exists.

Computed classically in this script (first principles: trial-division phi,
then gcd), the unique n in [1, 16] satisfying this is n = 9 (phi(9) = 6,
gcd(9, 6) = 3 != 1; the other odd n in range -- 1,3,5,7,11,13,15 -- are all
cyclic numbers). That makes it a unique-marked-item Grover search over 4
qubits (16-element space), index 8 (binary 1000).

THE QUANTUM CIRCUIT
--------------------
Standard Grover's algorithm: uniform superposition over 4 qubits, an oracle
that phase-flips the unique marked basis state |1000>, followed by the
standard diffusion (inversion-about-the-mean) operator, repeated the
optimal number of iterations for N=16, M=1 marked items
(floor(pi/4 * sqrt(N/M)) = 3), then measurement. Run on the ideal
AerSimulator (no noise model), 4096 shots.

PASS/FAIL
----------
The script prints PASS if the most frequently measured 4-bit outcome equals
the classically-computed marked index (binary for n-1 where n=9, i.e. 1000),
and FAIL otherwise.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no external tables)
# ---------------------------------------------------------------------------

def euler_phi(n: int) -> int:
    """Euler's totient via trial division, computed from scratch."""
    result = n
    p = 2
    m = n
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            result -= result // p
        p += 1
    if m > 1:
        result -= result // m
    return result


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def is_cyclic_number(n: int) -> bool:
    """n is a 'cyclic number' iff gcd(n, phi(n)) == 1 (OEIS A003277)."""
    return gcd(n, euler_phi(n)) == 1


N_QUBITS = 4
SPACE_SIZE = 2 ** N_QUBITS  # 16

marked_indices = []
for idx in range(SPACE_SIZE):
    n = idx + 1  # n ranges 1..16
    if n % 2 == 1 and not is_cyclic_number(n):
        marked_indices.append(idx)

assert len(marked_indices) == 1, (
    f"expected a unique marked item in [1,16], got {marked_indices}"
)
marked_index = marked_indices[0]
marked_n = marked_index + 1
print(f"Classical search: odd non-cyclic-number n in [1,16] -> n = {marked_n} "
      f"(phi({marked_n})={euler_phi(marked_n)}, "
      f"gcd({marked_n},{euler_phi(marked_n)})={gcd(marked_n, euler_phi(marked_n))})")
print(f"Marked index = {marked_index} (binary {format(marked_index, '04b')})")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for the unique marked index
# ---------------------------------------------------------------------------

def build_oracle(n_qubits: int, target_index: int) -> QuantumCircuit:
    """Phase-flip the |target_index> basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(target_index, f"0{n_qubits}b")
    # Flip 0-bits to 1 so we can use a multi-controlled Z on all-ones,
    # then flip back.
    for i, b in enumerate(reversed(bits)):
        if b == "0":
            qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, b in enumerate(reversed(bits)):
        if b == "0":
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


num_iterations = int(np.floor((np.pi / 4) * np.sqrt(SPACE_SIZE / 1)))
print(f"Grover iterations: {num_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(N_QUBITS, marked_index)
diffuser = build_diffuser(N_QUBITS)

for _ in range(num_iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: classical register bit c0 is the rightmost character.
best_outcome = max(counts, key=counts.get)
best_index = int(best_outcome, 2)
best_count = counts[best_outcome]

print(f"Measurement histogram (top 5): "
      f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Most frequent outcome: {best_outcome} (index {best_index}), "
      f"{best_count}/{shots} shots ({100 * best_count / shots:.1f}%)")


# ---------------------------------------------------------------------------
# 4. Compare quantum result to classical answer
# ---------------------------------------------------------------------------

verified = (best_index == marked_index)

if verified:
    print(f"PASS: Grover search found n = {best_index + 1}, matching the "
          f"classically-computed answer n = {marked_n}.")
else:
    print(f"FAIL: Grover search returned index {best_index} "
          f"(n={best_index + 1}), expected index {marked_index} (n={marked_n}).")
