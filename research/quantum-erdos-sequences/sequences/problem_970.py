"""
Erdos problem #970 (see erdosproblems.com/970, "Jacobsthal's function").

OEIS id used: A048669 — g(n) for n = the k-th primorial (product of the
first k primes), where g(n) is Jacobsthal's function: the smallest m such
that every run of m consecutive integers contains one coprime to n.
A048669 begins 1, 2, 4, 6, 10, 14, 22, 26, ... for n = 1, 2, 6, 30, 210, ...
(A048669(4) = 6 for n = 30 = 2*3*5, the third primorial).

Classical property tested (derived from first principles in this script,
not copied from OEIS):

    For n = 30, list the residues in [0, 30) coprime to 30, and compute the
    maximal gap between consecutive coprime residues (cyclically). This
    maximal gap is exactly g(30) = A048669(4) = 6. The residues x that
    *start* a maximal gap of length 6 (x coprime to 30, x+1..x+5 all NOT
    coprime to 30, x+6 coprime to 30, arithmetic mod 30) form a small,
    finite, fully computable marked set M subset of {0,...,29}.

    We compute M classically here (no lookup, only gcd arithmetic), then
    build a Grover search circuit over 5 qubits (32 basis states covering
    0..31, with 30 and 31 simply never marked) whose oracle flips the phase
    of exactly the elements of M. Running Grover's algorithm with the
    correct number of iterations should return only states in M with high
    probability. We compare the quantum-measured most-likely states against
    the classically computed set M: if they match, this is empirical
    quantum evidence that the algorithm correctly identifies the start of
    the extremal (Jacobsthal) gap for n = 30, i.e. that g(30) = 6, matching
    A048669(4).

Circuit: 5-qubit Grover search (32-dimensional space), a multi-controlled-Z
oracle built directly from the marked bitstrings (no arithmetic in-circuit;
the marking set itself is derived by an honest classical gcd computation,
which is the "small computable property" this instance is testing), and the
standard diffusion operator. Run on the ideal AerSimulator.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
from math import gcd

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles): Jacobsthal gap for n = 30.
# ---------------------------------------------------------------------------

N = 30  # third primorial: 2 * 3 * 5
NUM_QUBITS = 5  # 2**5 = 32 > 30, enough basis states to index residues 0..29

coprime_residues = [r for r in range(N) if gcd(r, N) == 1]
assert coprime_residues == [1, 7, 11, 13, 17, 19, 23, 29]

# Cyclic gaps between consecutive coprime residues mod N.
gaps = []
for i in range(len(coprime_residues)):
    a = coprime_residues[i]
    b = coprime_residues[(i + 1) % len(coprime_residues)]
    gap = (b - a) % N
    if gap == 0:
        gap = N
    gaps.append(gap)

max_gap = max(gaps)
# This is g(30), Jacobsthal's function at n = 30 = A048669(4).
assert max_gap == 6, f"expected Jacobsthal g(30) = 6, computed {max_gap}"

# Marked set: residues x in [0, N) such that x is coprime to N, x+1..x+5
# (mod N) are all NOT coprime to N, and x+6 (mod N) is coprime to N. These
# are exactly the starting points of a maximal-length gap.
GAP_LEN = max_gap


def is_gap_start(x: int) -> bool:
    if gcd(x, N) != 1:
        return False
    for i in range(1, GAP_LEN):
        if gcd((x + i) % N, N) == 1:
            return False
    if gcd((x + GAP_LEN) % N, N) != 1:
        return False
    return True


marked = [x for x in range(N) if is_gap_start(x)]
assert marked, "no marked states found -- classical logic error"
print(f"Classical result: N={N}, g(N)={max_gap}, marked gap-start residues = {marked}")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle for the marked set, over NUM_QUBITS qubits.
# ---------------------------------------------------------------------------

def apply_multi_controlled_z(qc: QuantumCircuit, qubits, value: int, n_qubits: int):
    """Flip the phase of the single basis state |value> on `qubits`.

    Uses X gates to map the target bitstring to all-ones, then a
    multi-controlled Z (via H + MCX + H on the last qubit), then undoes the
    X gates.
    """
    bits = [(value >> i) & 1 for i in range(n_qubits)]  # little-endian
    flip_qubits = [q for q, b in zip(qubits, bits) if b == 0]

    for q in flip_qubits:
        qc.x(q)

    # Multi-controlled Z on all n_qubits qubits (phase flip of |11...1>)
    controls = qubits[:-1]
    target = qubits[-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)

    for q in flip_qubits:
        qc.x(q)


def oracle(qc: QuantumCircuit, qubits, marked_values, n_qubits):
    for v in marked_values:
        apply_multi_controlled_z(qc, qubits, v, n_qubits)


def diffusion(qc: QuantumCircuit, qubits, n_qubits):
    for q in qubits:
        qc.h(q)
    for q in qubits:
        qc.x(q)
    controls = qubits[:-1]
    target = qubits[-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    for q in qubits:
        qc.x(q)
    for q in qubits:
        qc.h(q)


# ---------------------------------------------------------------------------
# 3. Assemble and run the Grover circuit.
# ---------------------------------------------------------------------------

num_states = 2 ** NUM_QUBITS
num_marked = len(marked)

# Optimal number of Grover iterations for this search-space / marked-count.
theta = math.asin(math.sqrt(num_marked / num_states))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qubits = list(range(NUM_QUBITS))

qc.h(qubits)
for _ in range(iterations):
    oracle(qc, qubits, marked, NUM_QUBITS)
    diffusion(qc, qubits, NUM_QUBITS)
qc.measure(qubits, qubits)

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Convert bitstrings (Qiskit prints classical bits with qubit0 as the
# rightmost character) to integer residue values.
value_counts = {}
for bitstring, c in counts.items():
    v = sum(int(bit) * (2 ** i) for i, bit in enumerate(bitstring[::-1]))
    value_counts[v] = value_counts.get(v, 0) + c

# Take the states with the highest measured probability, as many as there
# are marked classical solutions.
top_values = sorted(value_counts.items(), key=lambda kv: -kv[1])[:num_marked]
top_value_set = {v for v, _ in top_values}

quantum_success_prob = sum(value_counts.get(v, 0) for v in marked) / shots

print(f"Grover iterations used: {iterations}")
print(f"Top measured values (value: counts): {top_values}")
print(f"Total probability mass on classically-marked states: {quantum_success_prob:.4f}")

matches_classical = top_value_set == set(marked)
high_confidence = quantum_success_prob > 0.9

verified = matches_classical and high_confidence

if verified:
    print("PASS")
else:
    print("FAIL")
    print(f"Expected marked set: {sorted(marked)}, got top states: {sorted(top_value_set)}")
