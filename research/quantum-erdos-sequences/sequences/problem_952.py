"""
Erdos problem #952 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com clone):
  number: 952
  comments: "Gaussian moat problem"
  tags: ["number theory"]
  oeis: ["N/A"]   -- no OEIS sequence is attached to this problem.

Because problem #952 carries no OEIS id, there is no "sequence" to search or
verify a membership test against in the sense the other lanes in this library
use. This script is therefore a best-honest-effort quantum instance built
directly from the problem's own mathematical content (the Gaussian moat
problem: can you walk to infinity among the Gaussian primes taking steps of
bounded length?) rather than from an OEIS sequence, and that limitation is
recorded here explicitly rather than papered over.

Classical property tested (computed from first principles in this script,
not copied from any table):
  Fix the Gaussian prime p = 3 + 2i (norm 13; 3+2i is prime in Z[i] because
  its norm 13 is a rational prime). Consider the 8 candidate Gaussian
  integers obtained by taking the 8 unit steps of a chess-king move
  (dx, dy in {-1,0,1}, excluding (0,0)) scaled by step length L = 2, i.e.
  candidates c_k = p + 3*(dx_k, dy_k) for k = 0..7 in a fixed order.
  For each candidate we classically decide Gaussian-primality (a Gaussian
  integer a+bi is prime iff: (i) if a==0 or b==0, |a| or |b| is a rational
  prime == 3 mod 4, or (ii) a^2+b^2 is a rational prime). Exactly the
  candidates that are themselves Gaussian primes are "marked" (with L=3 for
  this p there happen to be exactly 2 marked out of 8, which keeps Grover's
  amplitude amplification in its well-conditioned minority-marked regime).
  This is a
  finite, computable predicate: "which of these 8 neighbours of p (at moat
  step length 2) are Gaussian primes" -- directly the local search structure
  the moat problem is about (whether p can keep stepping through primes).

Quantum computation:
  A 3-qubit Grover search over the 8 candidate indices, with a diffusion-based
  oracle built from the classical marked list, is used to amplify and then
  measure the marked (Gaussian-prime) indices. The quantum result (most
  frequent measured index/indices) is compared against the classical answer
  (the set of marked indices from direct classical primality testing).

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical Gaussian-integer primality test (first principles).
# ---------------------------------------------------------------------------

def is_rational_prime(n: int) -> bool:
    n = abs(n)
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def is_gaussian_prime(a: int, b: int) -> bool:
    """a + bi is a Gaussian prime iff:
       - a == 0: |b| is a rational prime with |b| % 4 == 3
       - b == 0: |a| is a rational prime with |a| % 4 == 3
       - else:   a^2 + b^2 is a rational prime
    (0 is not prime; units (+-1, +-i) are not prime.)
    """
    if a == 0 and b == 0:
        return False
    if a == 0:
        return is_rational_prime(b) and abs(b) % 4 == 3
    if b == 0:
        return is_rational_prime(a) and abs(a) % 4 == 3
    return is_rational_prime(a * a + b * b)


# ---------------------------------------------------------------------------
# 2. Build the small finite instance: 8 candidate neighbours of a fixed
#    Gaussian prime p = 3+2i, at moat step length L = 2.
# ---------------------------------------------------------------------------

p = (3, 2)
assert is_gaussian_prime(*p), "3+2i must itself be a Gaussian prime (norm 13)"

L = 3
directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
assert len(directions) == 8

candidates = [(p[0] + L * dx, p[1] + L * dy) for dx, dy in directions]
classical_marked = [is_gaussian_prime(a, b) for a, b in candidates]
marked_indices = [i for i, m in enumerate(classical_marked) if m]

print("Candidates (index: point -> gaussian-prime?):")
for i, ((a, b), m) in enumerate(zip(candidates, classical_marked)):
    print(f"  {i}: {a}{'+' if b >= 0 else ''}{b}i -> {m}")
print("Classically marked indices:", marked_indices)

assert 1 <= len(marked_indices) <= 7, (
    "Grover instance needs at least one marked and at least one unmarked "
    "item for a meaningful search; adjust p/L if this fires."
)


# ---------------------------------------------------------------------------
# 3. Grover search over the 8 indices (3 qubits) for the marked set.
# ---------------------------------------------------------------------------

n_qubits = 3  # 2^3 = 8 candidates


def oracle_circuit(marked, n):
    """Phase-flip oracle: applies -1 phase to each marked computational
    basis state (little-endian index), built as a multi-controlled Z per
    marked index using X-gates to remap 0-bits to controls."""
    qc = QuantumCircuit(n, name="Oracle")
    for idx in marked:
        bits = format(idx, f"0{n}b")[::-1]  # little-endian bit string
        flip_qubits = [q for q, bit in enumerate(bits) if bit == "0"]
        for q in flip_qubits:
            qc.x(q)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc


def diffusion_circuit(n):
    qc = QuantumCircuit(n, name="Diffusion")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


M = len(marked_indices)
N = 2 ** n_qubits
# Optimal number of Grover iterations for M marked out of N.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5)) if theta > 0 else 0

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))

oracle = oracle_circuit(marked_indices, n_qubits)
diffuser = diffusion_circuit(n_qubits)

for _ in range(iterations):
    qc.append(oracle.to_gate(), range(n_qubits))
    qc.append(diffuser.to_gate(), range(n_qubits))

qc.measure(range(n_qubits), range(n_qubits))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Convert little-endian bitstrings back to indices, and figure out which
# indices Grover amplified above a clear threshold.
index_counts = {}
for bitstring, c in counts.items():
    idx = int(bitstring[::-1], 2)  # qiskit bitstrings are big-endian in string form
    index_counts[idx] = index_counts.get(idx, 0) + c

# Expected per-marked-index share of shots if amplification worked well.
threshold = shots / N  # uniform-baseline count per index; amplified ones exceed it
quantum_found = sorted(i for i, c in index_counts.items() if c > threshold)

print("\nGrover iterations used:", iterations)
print("Measurement counts by candidate index:", dict(sorted(index_counts.items())))
print("Quantum-found (amplified) indices:", quantum_found)
print("Classical marked indices:         ", sorted(marked_indices))

passed = quantum_found == sorted(marked_indices)
print("\nPASS" if passed else "FAIL")
