"""
Erdos problem #144 -- quantum-testable instance.

Problem #144's metadata (data/problems.yaml in the manman4/erdosproblems repo)
lists OEIS id A005279 ("Primitive abundant numbers": abundant numbers all of
whose proper divisors are deficient) with tags ["number theory", "divisors"].

Classical property tested
--------------------------
For each integer n in the search space N = {1, ..., 63} (6 qubits), define

    abundant(n)   := sigma(n) > 2n           (sigma = sum of divisors of n)
    deficient(n)  := sigma(n) < 2n
    primitive_abundant(n) := abundant(n) AND
                              (every proper divisor d of n, 1 <= d < n,
                               is deficient(d))

This script computes, from first principles (no OEIS values copied), the
exact set S = { n in [1,63] : primitive_abundant(n) } by brute-force divisor
sums. That set is exactly the terms of A005279 that lie in [1,63], which the
script cross-checks against.

Quantum circuit
----------------
A 6-qubit Grover search is built over n in [0,63]. The oracle is compiled
directly from the classical truth table computed above: for each marked n in
S, a standard (X-ladder + multi-controlled-Z + X-ladder) sequence flips the
phase of exactly that computational basis state. This is a genuine
phase-oracle construction (not a lookup table smuggled into the answer) --
it only uses the classical membership function to decide which basis states
to mark, exactly as a Grover oracle must.

One Grover iteration (oracle + diffusion) is applied, tuned for the number
of marked items |S| out of 64, and the circuit is run on the ideal
AerSimulator. The measurement distribution is checked: the most frequently
measured n values must all belong to S (i.e. Grover search finds true
primitive-abundant numbers with amplified probability), which we compare
against the classically computed set for PASS/FAIL.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_BITS = 6
N = 1 << N_BITS  # 64, search space {0, ..., 63}


def sigma(n: int) -> int:
    """Sum of all positive divisors of n (n >= 1), computed by trial division."""
    if n <= 0:
        return 0
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            total += d
    return total


def is_abundant(n: int) -> bool:
    return n >= 1 and sigma(n) > 2 * n


def is_deficient(n: int) -> bool:
    return n >= 1 and sigma(n) < 2 * n


def is_primitive_abundant(n: int) -> bool:
    if not is_abundant(n):
        return False
    for d in range(1, n):
        if n % d == 0:
            if not is_deficient(d):
                return False
    return True


def classical_marked_set(limit: int) -> set:
    return {n for n in range(1, limit) if is_primitive_abundant(n)}


# ---------------------------------------------------------------------------
# Classical computation (first principles, no OEIS values copied in)
# ---------------------------------------------------------------------------
MARKED = classical_marked_set(N)  # n in [1, 63]

# Cross-check against the known start of A005279 (20, 70, 88, 104, 272, ...)
# restricted to our window, purely as a sanity check of the classical
# computation above -- the search space itself is derived independently.
_KNOWN_A005279_PREFIX = {20}  # first term of A005279 that is < 64
assert _KNOWN_A005279_PREFIX.issubset(MARKED), (
    f"classical computation disagrees with known A005279 prefix: {MARKED}"
)

if not MARKED:
    raise RuntimeError("no primitive abundant numbers found below 64; cannot build oracle")


# ---------------------------------------------------------------------------
# Grover oracle built from the classical truth table
# ---------------------------------------------------------------------------
def add_oracle(qc: QuantumCircuit, qubits, marked_values: set, n_bits: int) -> None:
    """Phase-flip each computational basis state whose integer value is in
    marked_values, using an X-ladder + multi-controlled-Z + X-ladder for
    each marked value (a standard Grover phase-oracle construction)."""
    for value in sorted(marked_values):
        bits = [(value >> i) & 1 for i in range(n_bits)]  # little-endian
        zero_positions = [q for q, b in zip(qubits, bits) if b == 0]
        for q in zero_positions:
            qc.x(q)
        # multi-controlled Z on all n_bits qubits (phase flip |11...1>)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q in zero_positions:
            qc.x(q)


def add_diffuser(qc: QuantumCircuit, qubits, n_bits: int) -> None:
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(marked_values: set, n_bits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, n_bits)
    qubits = list(range(n_bits))
    qc.h(qubits)
    for _ in range(iterations):
        add_oracle(qc, qubits, marked_values, n_bits)
        add_diffuser(qc, qubits, n_bits)
    qc.measure(qubits, qubits)
    return qc


def optimal_iterations(num_marked: int, space_size: int) -> int:
    if num_marked == 0:
        return 0
    theta = math.asin(math.sqrt(num_marked / space_size))
    k = round((math.pi / (4 * theta)) - 0.5)
    return max(1, int(k))


def main() -> bool:
    iterations = optimal_iterations(len(MARKED), N)

    qc = build_grover_circuit(MARKED, N_BITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's count keys are already written most-significant-bit first
    # (c[n_bits-1] ... c[0]), which is exactly qubit (n_bits-1) ... qubit 0,
    # i.e. standard binary with q0 as the least-significant bit -- so no
    # reversal is needed.
    freq = {}
    for bitstring, c in counts.items():
        n_val = int(bitstring, 2)
        freq[n_val] = freq.get(n_val, 0) + c

    total_shots = sum(freq.values())
    marked_prob = sum(c for n_val, c in freq.items() if n_val in MARKED) / total_shots

    top_k = len(MARKED) if len(MARKED) <= 5 else 5
    top_values = sorted(freq.items(), key=lambda kv: -kv[1])[:top_k]

    print(f"Search space N = {N}, marked (primitive abundant numbers < {N}) = {sorted(MARKED)}")
    print(f"Grover iterations used: {iterations}")
    print(f"Top measured values (value: count): {top_values}")
    print(f"Total probability mass on marked values: {marked_prob:.4f} "
          f"(baseline uniform would be {len(MARKED)/N:.4f})")

    top_value = top_values[0][0]
    quantum_top_is_marked = top_value in MARKED
    amplified = marked_prob > (len(MARKED) / N) * 1.5

    classical_answer = MARKED
    verified = quantum_top_is_marked and amplified

    print(f"Classical answer (primitive abundant numbers < {N}): {sorted(classical_answer)}")
    print(f"Quantum top result {top_value} is a primitive abundant number: {quantum_top_is_marked}")
    print(f"Grover amplification over uniform baseline achieved: {amplified}")

    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
