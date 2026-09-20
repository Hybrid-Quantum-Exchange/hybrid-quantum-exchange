"""
Erdos problem #969 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 969"):
    oeis: ["A013928"]
    tags: ["number theory"]

OEIS A013928 is the sequence a(n) = number of nonsquarefree numbers <= n
(equivalently n - Q(n), where Q(n) is the count of squarefree numbers up to
n). A positive integer k is squarefree iff no prime square p^2 divides k.

Classical property tested here (computed from first principles, not looked
up from OEIS): for N = 16, identify the exact subset of {1, ..., N} that is
NOT squarefree (i.e. divisible by some p^2), and its size a(16). This is a
small, fully finite/computable search problem: "which of the 16 integers in
[1,16] are non-squarefree?" -- exactly the quantity A013928 counts.

Quantum approach: Grover search over a 4-qubit register encoding integers
0..15 (representing 1..16 via k = index + 1). The oracle phase-flips exactly
the basis states corresponding to non-squarefree k, built directly from the
classical squarefree test (no lookup table smuggled in from OEIS -- the
oracle is derived from the same is_squarefree() function used for the
classical answer). We run the standard Grover diffusion for the computed
optimal number of iterations on the ideal AerSimulator, then check that the
states we measure with high probability are exactly (a subset of) the
non-squarefree numbers, and that the full marked set found this way (by
sampling) matches the classical set. We also cross-check a(16) computed
classically two independent ways (direct trial division, and via the count
of squarefree numbers) to make sure the classical answer itself is not a
typo.

PASS criterion: (1) the two classical computations of a(16) agree, and
(2) every one of the top-|marked| measured outcomes from the Grover circuit
(ranked by measured probability) is a genuine non-squarefree number in
[1,16], and the marked set reconstructed from the top outcomes equals the
classical non-squarefree set exactly.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, two independent methods)
# ---------------------------------------------------------------------------

N = 16  # instance size: test integers 1..16, encoded on 4 qubits (0..15)
NUM_QUBITS = 4
assert 2 ** NUM_QUBITS == N


def is_squarefree(k: int) -> bool:
    """True iff no prime square divides k. Trial division from scratch."""
    if k < 1:
        raise ValueError("k must be positive")
    n = k
    p = 2
    while p * p <= n:
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
        p += 1
    return True


def sieve_squarefree(limit: int):
    """Independent method: sieve out multiples of p^2 for every prime p."""
    flags = [True] * (limit + 1)  # flags[k] = True means "still squarefree"
    flags[0] = False
    p = 2
    while p * p <= limit:
        # simple primality check for p (limit is tiny, this is fine)
        is_prime = p >= 2 and all(p % d != 0 for d in range(2, int(p ** 0.5) + 1))
        if is_prime:
            for multiple in range(p * p, limit + 1, p * p):
                flags[multiple] = False
        p += 1
    return flags  # flags[k] True iff k is squarefree, for k=1..limit


# Method A: direct trial division per integer.
nonsquarefree_A = [k for k in range(1, N + 1) if not is_squarefree(k)]

# Method B: sieve of p^2-multiples.
flags = sieve_squarefree(N)
nonsquarefree_B = [k for k in range(1, N + 1) if not flags[k]]

assert nonsquarefree_A == nonsquarefree_B, (
    f"classical methods disagree: {nonsquarefree_A} vs {nonsquarefree_B}"
)

NONSQUAREFREE = nonsquarefree_A          # the classical answer: e.g. [4, 8, 9, 12, 16]
A_OF_16 = len(NONSQUAREFREE)             # this is OEIS A013928(16)
MARKED_INDICES = sorted(k - 1 for k in NONSQUAREFREE)  # 0-based, for the qubit register

print(f"Classical: non-squarefree numbers in [1,{N}] = {NONSQUAREFREE}")
print(f"Classical: A013928({N}) = a(16) = {A_OF_16}")


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the non-squarefree indices
# ---------------------------------------------------------------------------

def build_oracle(num_qubits: int, marked_indices) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{num_qubits}b")  # MSB..LSB over qubits[n-1..0]
        # flip qubits that should be 0 in this index, so the all-ones pattern
        # corresponds to |idx>
        zero_qubits = [num_qubits - 1 - i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
            qc.h(num_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


oracle = build_oracle(NUM_QUBITS, MARKED_INDICES)
diffuser = build_diffuser(NUM_QUBITS)

M = len(MARKED_INDICES)
# optimal number of Grover iterations for M marked out of N
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

print(f"Grover: N={N}, M={M} marked, iterations={iterations}")


# ---------------------------------------------------------------------------
# 3. Run on ideal AerSimulator
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 20000
tqc = transpile(qc, basis_gates=["u3", "cx", "x", "h", "z", "cz"])
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: classical bit string is c[n-1]...c[0]; our circuit
# used qubit i <-> classical bit i, and measurement strings from Qiskit are
# printed most-significant-qubit first == qubit (n-1) ... qubit 0, which
# matches the index encoding used in build_oracle (MSB..LSB over
# qubits[n-1..0]). So int(bitstring, 2) directly gives the encoded index.
sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_outcomes = [int(bitstr, 2) for bitstr, _ in sorted_counts[:M]]

measured_numbers = sorted(idx + 1 for idx in top_outcomes)
print(f"Quantum: top {M} measured outcomes (as numbers) = {measured_numbers}")
print(f"Quantum: measurement counts (top {M}) = {sorted_counts[:M]}")


# ---------------------------------------------------------------------------
# 4. Verify against the classical answer
# ---------------------------------------------------------------------------

classical_ok = nonsquarefree_A == nonsquarefree_B
top_are_all_nonsquarefree = all(n in NONSQUAREFREE for n in measured_numbers)
sets_match = measured_numbers == NONSQUAREFREE

verified = classical_ok and top_are_all_nonsquarefree and sets_match

print(f"Check: classical cross-check agrees = {classical_ok}")
print(f"Check: all top-{M} measured outcomes are non-squarefree = {top_are_all_nonsquarefree}")
print(f"Check: measured set == classical A013928 support set = {sets_match}")

if verified:
    print("PASS")
else:
    print("FAIL")
