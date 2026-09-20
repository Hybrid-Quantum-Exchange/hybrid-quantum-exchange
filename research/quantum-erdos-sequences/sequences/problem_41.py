"""
Erdos problem #41 (see https://github.com/manman4/erdosproblems,
data/problems.yaml, number: "41") — tags: number theory, Sidon sets,
additive combinatorics. Prize $500, status: open.

Limitation, stated honestly up front: problem #41's entry in problems.yaml
carries oeis: ["N/A"] — there is no OEIS sequence id attached to this
problem, so this script cannot test membership in an OEIS-indexed
sequence. Instead it tests a small, finite, genuinely-Sidon-set property
that is exactly the combinatorial object problem #41 is about, and that a
Grover search circuit can search over honestly.

Classical property under test
------------------------------
A Sidon set (also called a B2 set) is a set of integers whose pairwise
sums (including a+a) are all distinct. The classical base set {0, 1, 3}
is a Sidon set of size 3. We ask: for which x in {0, 1, ..., 7} (i.e.
representable by 3 bits) does adding x turn {0, 1, 3} into a Sidon set
of size 4, i.e. {0, 1, 3, x} is a Sidon set?

This script first computes the answer by brute force from first
principles (checking every pairwise sum of every candidate x for
duplicates), then builds a 3-qubit Grover search circuit whose oracle
marks exactly the good x values, and runs it on the ideal AerSimulator.
Since brute force finds exactly one good x in {0,...,7}, this is a
textbook single-marked-item Grover instance (N=8, 1 iteration is
optimal), so the quantum search should return that x with high
probability. We compare the quantum result to the classical answer and
print PASS/FAIL.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation: find the good x values from first principles.
# ---------------------------------------------------------------------------

def is_sidon(s):
    """Return True iff all pairwise sums a+b (a<=b, a,b in s) are distinct."""
    sums = []
    for i in range(len(s)):
        for j in range(i, len(s)):
            v = s[i] + s[j]
            if v in sums:
                return False
            sums.append(v)
    return True


BASE = [0, 1, 3]
N_BITS = 3
N = 2 ** N_BITS  # search space size = 8

classical_good = []
for x in range(N):
    if x in BASE:
        continue
    if is_sidon(BASE + [x]):
        classical_good.append(x)

print(f"Classical brute-force search over x in 0..{N - 1}:")
print(f"  base Sidon set: {BASE}")
print(f"  x making {{0,1,3,x}} a Sidon set: {classical_good}")

if len(classical_good) != 1:
    raise SystemExit(
        f"Expected exactly one marked item for a clean single-item Grover "
        f"demo, found {classical_good}. Refusing to fake a pass."
    )

target = classical_good[0]
print(f"  -> unique classical answer: x = {target} "
      f"(binary {format(target, '0{}b'.format(N_BITS))})")


# ---------------------------------------------------------------------------
# 2. Quantum: Grover search over the 3-bit register for x == target.
# ---------------------------------------------------------------------------

def build_oracle(n_bits, marked):
    """Phase-flip oracle marking the single basis state `marked`."""
    qc = QuantumCircuit(n_bits, name="Oracle")
    bits = format(marked, f"0{n_bits}b")[::-1]  # little-endian qubit order
    # Flip qubits that should be 0 in the marked state, so the marked
    # state maps to |11...1>, then apply a multi-controlled Z, then
    # flip back.
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    return qc


def build_diffuser(n_bits):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_bits, name="Diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


oracle = build_oracle(N_BITS, target)
diffuser = build_diffuser(N_BITS)

# Optimal number of Grover iterations for N=8, M=1 marked item:
# r ~ floor(pi/4 * sqrt(N/M)) = floor(pi/4 * sqrt(8)) = 2
iterations = int(np.floor((np.pi / 4) * np.sqrt(N / 1)))
iterations = max(iterations, 1)
print(f"  Grover iterations used: {iterations}")

qc = QuantumCircuit(N_BITS, N_BITS)
qc.h(range(N_BITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_BITS), range(N_BITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

print("Measurement counts:", counts)

# Qiskit reports bitstrings as qubit[n-1]...qubit[0]; convert back to int
# with qubit 0 as the least-significant bit (matches our little-endian
# encoding above).
best_bitstring = max(counts, key=counts.get)
quantum_answer = int(best_bitstring[::-1], 2)
confidence = counts[best_bitstring] / shots

print(f"Most frequent measured value: x = {quantum_answer} "
      f"(confidence {confidence:.3f} over {shots} shots)")


# ---------------------------------------------------------------------------
# 3. Compare and report.
# ---------------------------------------------------------------------------

verified = (quantum_answer == target) and (confidence > 0.5)

print()
if verified:
    print(f"PASS: quantum Grover search found x = {quantum_answer}, "
          f"matching the classical Sidon-set-completion answer x = {target}.")
else:
    print(f"FAIL: quantum result x = {quantum_answer} "
          f"(confidence {confidence:.3f}) does not match classical "
          f"answer x = {target}.")
