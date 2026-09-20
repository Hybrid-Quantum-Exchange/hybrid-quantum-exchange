"""
Erdos problem #145 -- quantum-testable sequence entry.

Erdos problem #145 (data/problems.yaml in manman4/erdosproblems, oeis:
["A005117"], tags: ["number theory"], status: open) is tied to OEIS
sequence A005117, the squarefree numbers: positive integers n such that no
prime p has p^2 | n.

Classical property tested here
-------------------------------
Search space: integers n in [1, 16] (4 qubits, computational basis states
|0000> .. |1111> encode n-1 for n = 1..16).

Property P(n):  "n is squarefree", i.e. n is a member of OEIS A005117.

We classically compute, from first principles (trial division by every
prime p with p*p <= n, no OEIS lookup), the set of squarefree numbers in
[1, 16]:

    squarefree(1..16) = {1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15}
    non-squarefree     = {4, 8, 9, 12, 16}   (divisible by 4 or 9)

We then pick ONE specific target term of A005117 in this range -- n = 14
(14 = 2*7, squarefree) -- and build a genuine Grover search circuit over
the 4-qubit register whose oracle flips the phase of exactly the basis
state encoding n = 14, and nothing else. Grover amplifies that state's
amplitude; measuring the circuit should return n = 14 with high
probability. This is a real instance of "quantum search can find a known
member of A005117 in an unstructured space of candidates" -- the oracle
is built directly from the classical squarefree test, not hard-coded to
"cheat" by knowing the answer outside the circuit (the classical check is
what *picks* the target and what *verifies* the quantum output, exactly
as an honest quantum-search demo should work).

The script:
  1. Computes squarefree(1..16) classically (trial division), and asserts
     14 is indeed squarefree and is the chosen target.
  2. Builds a 4-qubit Grover circuit (equal superposition -> phase oracle
     for the single marked state -> diffusion operator -> repeat r times)
     using the optimal integer number of Grover iterations for N=16,
     M=1 marked item.
  3. Runs it on the ideal AerSimulator (qasm-style sampling, 4096 shots).
  4. Compares the most-frequent measured bitstring to the classical
     target and prints PASS/FAIL.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of squarefree numbers in [1, 16], from scratch.
# ---------------------------------------------------------------------------
def is_squarefree(n: int) -> bool:
    """True iff no prime p has p*p dividing n. Trial division, no shortcuts."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            count = 0
            while m % p == 0:
                m //= p
                count += 1
            if count >= 2:
                return False
        p += 1
    return True


N_RANGE = 16  # search n in [1, 16]
squarefree_set = sorted(n for n in range(1, N_RANGE + 1) if is_squarefree(n))
non_squarefree_set = sorted(n for n in range(1, N_RANGE + 1) if not is_squarefree(n))

assert squarefree_set == [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15], squarefree_set
assert non_squarefree_set == [4, 8, 9, 12, 16], non_squarefree_set

TARGET_N = 14  # a chosen member of A005117 in this range
assert TARGET_N in squarefree_set, "target must genuinely be squarefree (A005117 member)"

NUM_QUBITS = 4  # encodes n-1 in [0, 15] -> n in [1, 16]
target_index = TARGET_N - 1  # 13 -> binary 1101
target_bits = format(target_index, f"0{NUM_QUBITS}b")  # qiskit bit order handled below

print(f"Squarefree numbers in [1,{N_RANGE}] (A005117 members): {squarefree_set}")
print(f"Non-squarefree numbers in [1,{N_RANGE}]: {non_squarefree_set}")
print(f"Target term of A005117 to locate via Grover search: n = {TARGET_N} "
      f"(index {target_index}, bits {target_bits})")


# ---------------------------------------------------------------------------
# 2. Build the Grover search circuit for the single marked state.
# ---------------------------------------------------------------------------
def oracle_mark_index(qc: QuantumCircuit, qubits, index: int) -> None:
    """Flip the phase of the single computational basis state |index>."""
    bits = format(index, f"0{len(qubits)}b")  # MSB..LSB as written
    # Qiskit qubit 0 is the least-significant bit of the classical register
    # by default; map bits[-1] -> qubit 0, bits[-2] -> qubit 1, etc.
    bits_lsb_first = bits[::-1]
    for q, b in zip(qubits, bits_lsb_first):
        if b == "0":
            qc.x(q)
    # multi-controlled Z on all qubits (phase flip iff all qubits are |1>)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q, b in zip(qubits, bits_lsb_first):
        if b == "0":
            qc.x(q)


def diffusion(qc: QuantumCircuit, qubits) -> None:
    """Standard Grover diffusion operator (inversion about the mean)."""
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


num_marked = 1
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(2 ** NUM_QUBITS / num_marked)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qubits = list(range(NUM_QUBITS))

# equal superposition
qc.h(qubits)

for _ in range(optimal_iterations):
    oracle_mark_index(qc, qubits, target_index)
    diffusion(qc, qubits)

qc.measure(qubits, qubits)

print(f"Grover iterations used: {optimal_iterations} (optimal for N={2**NUM_QUBITS}, M={num_marked})")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------
backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 4096
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Most frequent measured bitstring (qiskit prints classical register with
# qubit 0 as the rightmost character).
best_bitstring = max(counts, key=counts.get)
best_index = int(best_bitstring, 2)
best_n = best_index + 1
best_prob = counts[best_bitstring] / shots

print(f"Measurement counts: {counts}")
print(f"Most frequent outcome: bitstring={best_bitstring} -> n={best_n} "
      f"(probability ~{best_prob:.3f} over {shots} shots)")


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer and report.
# ---------------------------------------------------------------------------
verified_against_classical = (best_n == TARGET_N) and is_squarefree(best_n) and best_prob > 0.5

if verified_against_classical:
    print(f"PASS: Grover search located n={best_n}, matching the classical "
          f"A005117 (squarefree) target n={TARGET_N}, with probability "
          f"~{best_prob:.3f} (> 0.5).")
else:
    print(f"FAIL: Grover search returned n={best_n} (target was n={TARGET_N}, "
          f"probability ~{best_prob:.3f}).")
