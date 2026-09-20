"""
Erdos problem #396 -- quantum-testable instance.

OEIS: A375077.

Erdos problem #396 asks: is it true that for every k there exists n such that
    prod_{0<=i<=k} (n-i)  divides  C(2n, n)  (the central binomial coefficient)?

A375077(k) is defined as the smallest such n for each k. A375077(1) = 2: for
k = 1 the product is n*(n-1), and n = 2 is the smallest n >= 1 for which
n*(n-1) divides C(2n, n) (product = 2, C(4,2) = 6, and 2 | 6).

Classical property tested here (k = 1, search space n in [0, 63]):
    n*(n-1) divides C(2n, n)
Exhaustive classical search over n = 0..63 (computed in this script, from
scratch, with Python's exact integer arithmetic and math.comb) finds exactly
one solution in that range: n = 2. This matches the published OEIS value
A375077(1) = 2 (that OEIS value is not taken on faith -- it is re-derived
here classically before the quantum circuit is trusted).

Quantum circuit: since the search space [0, 63] has a UNIQUE marked element
(n = 2) among N = 64 = 2^6 candidates, this is a textbook unstructured-search
instance for Grover's algorithm. We build:
  - a 6-qubit register encoding n in binary (n = q5 q4 q3 q2 q1 q0, standard
    Qiskit little-endian qubit ordering),
  - an oracle that flips the phase of the |000010> state (n = 2, i.e. only
    qubit 1 is |1>) using a multi-controlled Z (with X-gates on the qubits
    that must be 0),
  - the standard Grover diffusion operator,
  - ceil(pi/4 * sqrt(N)) ~ 1 iteration for N = 64 (optimal integer count is
    computed exactly below),
run on Qiskit Aer's ideal statevector-based AerSimulator, then measure and
compare the most frequently measured bitstring to the classical answer.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from math import comb

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed from first principles, no OEIS lookup).
# ---------------------------------------------------------------------------

def classical_search(n_max: int):
    """Return all n in [0, n_max] with n*(n-1) | C(2n, n)."""
    hits = []
    for n in range(0, n_max + 1):
        val = n * (n - 1)
        if val == 0:
            continue
        if comb(2 * n, n) % val == 0:
            hits.append(n)
    return hits


NUM_QUBITS = 6
N = 2 ** NUM_QUBITS  # 64 candidates: n = 0 .. 63

classical_hits = classical_search(N - 1)
assert classical_hits == [2], (
    f"Expected a unique classical solution n=2 in [0,{N-1}], got {classical_hits}"
)
target = classical_hits[0]
target_bits = format(target, f"0{NUM_QUBITS}b")  # MSB..LSB string, e.g. "000010"

print(f"Classical search over n in [0,{N-1}] for n*(n-1) | C(2n,n):")
print(f"  unique solution n = {target}  (matches OEIS A375077(1) = 2)")
print(f"  target bitstring (MSB..LSB) = {target_bits}")


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffusion for the unique marked state |target>.
# ---------------------------------------------------------------------------

def apply_oracle(qc: QuantumCircuit, qubits, target_int: int, n_qubits: int):
    """Phase-flip the computational basis state encoding target_int.

    Qiskit uses little-endian: qubits[0] is the least-significant bit.
    """
    bits = format(target_int, f"0{n_qubits}b")[::-1]  # bits[i] = value of qubits[i]
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]

    for i in zero_positions:
        qc.x(qubits[i])

    # Multi-controlled Z on all n_qubits: flip phase of |11...1>.
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])

    for i in zero_positions:
        qc.x(qubits[i])


def apply_diffusion(qc: QuantumCircuit, qubits, n_qubits: int):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


# Optimal number of Grover iterations for 1 marked item out of N.
theta = math.asin(1.0 / math.sqrt(N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qubits = list(range(NUM_QUBITS))

qc.h(qubits)  # uniform superposition over all 64 candidate n values

for _ in range(iterations):
    apply_oracle(qc, qubits, target, NUM_QUBITS)
    apply_diffusion(qc, qubits, NUM_QUBITS)

qc.measure(qubits, qubits)

print(f"\nGrover circuit: {NUM_QUBITS} qubits, N={N}, iterations={iterations}")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator(method="statevector")
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-register bitstrings are printed MSB..LSB already
# (cN-1 ... c1 c0), matching our target_bits convention (qubits[0]=LSB
# appears as the rightmost character).
best_bitstring = max(counts, key=counts.get)
best_count = counts[best_bitstring]
best_int = int(best_bitstring, 2)

print(f"\nTop measured outcome: {best_bitstring} (n={best_int}), "
      f"{best_count}/{shots} shots ({100*best_count/shots:.1f}%)")
print(f"Target bitstring:     {target_bits} (n={target})")

quantum_matches_classical = (best_int == target)


# ---------------------------------------------------------------------------
# 4. Verdict.
# ---------------------------------------------------------------------------

if quantum_matches_classical:
    print("\nPASS: Grover search found the unique n with n*(n-1) | C(2n,n), "
          "matching the classical result and OEIS A375077(1) = 2.")
else:
    print("\nFAIL: Grover search result did not match the classical answer.")
