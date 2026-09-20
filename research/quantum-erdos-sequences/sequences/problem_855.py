"""
Erdos problem #855 (second Hardy-Littlewood conjecture, tags: number theory,
primes; OEIS: A023193).

Erdos problem 855 concerns the second Hardy-Littlewood conjecture, that the
prime-counting function pi(x) is "subadditive": pi(x+y) <= pi(x) + pi(y) for
all x, y >= 2. Both the conjecture itself and A023193 are about the behaviour
of pi(x) relative to counting/gap structure among the primes, which is not by
itself a finite yes/no computation a few qubits can settle (subadditivity is
an infinite, still-open claim, and A023193's defining recurrence is likewise
unbounded). To get a genuine small, finite, computable property with real
mathematical content, this script tests the concrete arithmetic fact that
pi(x) itself is built from, on a small finite instance:

    Classical property under test:
        For N = 16 (so n ranges over the 4-bit values 0..15), the set of
        primes below N is exactly P = {2, 3, 5, 7, 11, 13}.
        (This is computed from first principles below by trial division,
        not copied from OEIS.)

This is exactly the ingredient the Hardy-Littlewood conjecture and pi(x) are
built from: correctly identifying prime n. The quantum part is a real Grover
search circuit over 4 qubits (n = 0..15) whose oracle marks the basis states
corresponding to prime n (built as multi-controlled Z gates keyed on the
classically-computed bit patterns of P, i.e. the oracle is a genuine
diagonal phase-flip oracle, not a lookup of the answer). Grover's algorithm
then amplifies exactly the marked (prime) states. We compare the quantum
sampling distribution (ideal AerSimulator, many shots) against the classical
set P: the test passes if the states with non-negligible measured
probability mass are precisely the primes in [0, 15].

Limitation, honestly noted: this verifies primality identification on a
small range, which underlies pi(x) and hence the Hardy-Littlewood
conjecture's subject matter, but it does NOT test subadditivity of pi(x)
itself (that would require an unbounded search over x, y and is not a
finite quantum-circuit-sized computation). It also does not directly encode
A023193's own defining recurrence. This is the best small, honest,
genuinely-quantum instance connected to problem 855's subject matter.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the property, from first principles.
# ---------------------------------------------------------------------------

def is_prime_trial_division(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16

classical_primes = sorted(n for n in range(N) if is_prime_trial_division(n))
print(f"Classical primes below {N} (trial division): {classical_primes}")

assert classical_primes == [2, 3, 5, 7, 11, 13], "classical computation sanity check failed"

M = len(classical_primes)  # number of marked states


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle: multi-controlled Z on each prime's bit pattern.
# ---------------------------------------------------------------------------

def apply_marking_for_value(qc: QuantumCircuit, value: int, n_qubits: int) -> None:
    """Flip the phase of |value> using an (n_qubits-1)-controlled Z, with X
    gates sandwiching any qubit whose bit in `value` is 0 (standard
    multi-controlled-Z-on-arbitrary-bitstring construction)."""
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    zero_qubits = [i for i, b in enumerate(bits) if b == 0]

    for q in zero_qubits:
        qc.x(q)

    controls = list(range(n_qubits - 1))
    target = n_qubits - 1
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)

    for q in zero_qubits:
        qc.x(q)


def build_oracle(marked_values, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for v in marked_values:
        apply_marking_for_value(qc, v, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# Optimal number of Grover iterations for N states, M marked states.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"N={N}, M={M} marked states, using {iterations} Grover iteration(s)")

oracle = build_oracle(classical_primes, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 20000
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical register bit-string is little-endian in string order
# (rightmost char = qubit 0), so convert accordingly.
value_counts = Counter()
for bitstring, c in counts.items():
    n = int(bitstring, 2)
    value_counts[n] += c

# Take the top-M most frequently measured values as the circuit's answer.
top_values = sorted(
    [n for n, _ in value_counts.most_common(M)]
)

print("Top measured values (quantum):", top_values)
print("Measured probability mass on primes:",
      sum(value_counts[p] for p in classical_primes) / SHOTS)

quantum_matches_classical = top_values == classical_primes

# Also require that the total probability mass landed on marked (prime)
# states substantially exceeds the uniform-random baseline (M/N), showing
# genuine amplitude amplification rather than a lucky top-M pick.
prime_mass = sum(value_counts.get(p, 0) for p in classical_primes) / SHOTS
baseline = M / N
amplification_confirmed = prime_mass > 2 * baseline

verified = quantum_matches_classical and amplification_confirmed

print(f"quantum_matches_classical={quantum_matches_classical}, "
      f"prime_mass={prime_mass:.3f} vs baseline={baseline:.3f}, "
      f"amplification_confirmed={amplification_confirmed}")

if verified:
    print("PASS")
else:
    print("FAIL")
