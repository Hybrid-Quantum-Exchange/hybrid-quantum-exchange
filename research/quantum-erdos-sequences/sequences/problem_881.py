"""
Erdos problem #881 -- quantum-testable instance.

Source metadata (from erdosproblems.com's data, data/problems.yaml in the
manman4/erdosproblems clone, entry "number: '881'"):
    tags: ["number theory", "additive basis"]
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #881 carries no OEIS sequence
id in the source data (oeis: ["N/A"]), so there is no published integer
sequence to target directly. What #881 *does* give us is a topic tag,
"additive basis", which is a well-defined, small, finite, computable notion:
a set S of non-negative integers is an additive basis of order h for a range
[0, N) if every integer in [0, N) can be written as a sum of at most h
elements of S (repetition/choice of subset allowed, depending on the exact
flavor). This script does not invent a fake OEIS value; instead it builds a
genuine finite decision/search instance in the spirit of the tag and checks
it both classically and with a real Grover-search quantum circuit.

Classical property being tested
--------------------------------
Let S = {1, 2, 4, 8} (a canonical minimal additive basis: distinct subsets
of S already realize every integer in [0, 15] exactly once as a subset sum,
i.e. S is an additive/subset basis for the interval [0, 15]).

For a fixed target t = 11, define the search space as all 2^4 = 16 subsets
of S, indexed by a 4-bit string b3 b2 b1 b0 (bit i selects whether the i-th
element of S, i.e. 2^i, is included). The predicate is:

    f(b) = 1   iff   subset_sum(b) == t

where subset_sum(b) = sum_{i : b_i = 1} 2^i, i.e. subset_sum(b) is literally
the integer whose binary representation is b. So f(b) = 1 iff b (interpreted
as a binary number) equals t. Because {1,2,4,8} is a perfect additive basis
of order <=4 for [0,15], this subset-sum predicate has exactly ONE solution
for any t in [0,15]: the standard textbook setup for Grover's algorithm with
a single marked item.

The script:
  1. Computes the classical answer for t = 11 by brute-force enumeration of
     all 16 subsets of S (first principles, no OEIS lookup).
  2. Builds a real Qiskit Grover search circuit: 4 qubits encode the subset
     index; the oracle is a multi-controlled-Z (built from standard X/MCZ
     gates) that phase-flips exactly the computational basis state |t>,
     which is correct precisely because subset_sum(b) == t iff b == t in
     this basis-{1,2,4,8} encoding; the diffuser is the standard Grover
     diffusion operator. Runs on the ideal AerSimulator (statevector +
     shots).
  3. Compares the most frequent measured bitstring against the classical
     answer and prints PASS/FAIL.

This is a genuine (if small) instance of a Grover search over an additive
structure related to the "additive basis" tag of problem #881, computed and
verified from first principles in this file. It is not a restatement of a
literal OEIS term, because problem #881 has none.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no lookups).
# ---------------------------------------------------------------------------

S = [1, 2, 4, 8]          # candidate additive basis for [0, 15]
N = 16                     # range covered: 0 .. 15
TARGET = 11                # t in [0, N)
N_QUBITS = 4                # log2(2^|S|) = |S|, one bit per element of S


def subset_sum(bits):
    """bits: tuple of 0/1 of length len(S), bit i selects S[i]."""
    return sum(S[i] for i in range(len(S)) if bits[i])


def verify_additive_basis(elements, n):
    """Check elements form an additive (subset-sum) basis for [0, n)."""
    reachable = set()
    for r in range(len(elements) + 1):
        for combo in itertools.combinations(elements, r):
            reachable.add(sum(combo))
    return all(v in reachable for v in range(n))


assert verify_additive_basis(S, N), "S must be an additive basis for [0, N)"

# Brute-force classical search for the subset(s) of S summing to TARGET.
classical_solutions = []
for b in itertools.product([0, 1], repeat=N_QUBITS):
    if subset_sum(b) == TARGET:
        classical_solutions.append(b)

assert len(classical_solutions) == 1, (
    f"expected a unique solution for target {TARGET}, got {classical_solutions}"
)
classical_bits = classical_solutions[0]
# Qiskit bit ordering: qubit 0 is the least-significant (rightmost) bit of
# the measured bitstring. bits[i] multiplies S[i] = 2**i, so bits[i] IS
# qubit i's value, and the marked integer equals TARGET itself.
classical_answer_int = TARGET
classical_answer_bitstring = format(TARGET, f"0{N_QUBITS}b")  # MSB..LSB, b3b2b1b0

print(f"Erdos problem #881 -- additive basis instance")
print(f"S = {S}, N = {N}, target t = {TARGET}")
print(f"Classical brute-force unique solution (subset bits, LSB=S[0]): {classical_bits}")
print(f"=> marked computational basis state |{classical_answer_bitstring}> "
      f"(integer {classical_answer_int})")


# ---------------------------------------------------------------------------
# 2. Quantum Grover search circuit.
# ---------------------------------------------------------------------------

def build_oracle(n_qubits, marked_int):
    """Phase-flip circuit marking the single computational basis state
    |marked_int> (little-endian: qubit 0 = LSB), via X-sandwiched MCZ."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    bits = format(marked_int, f"0{n_qubits}b")[::-1]  # bits[i] -> qubit i
    zero_qubits = [i for i, bit in enumerate(bits) if bit == "0"]

    for q in zero_qubits:
        qc.x(q)

    # multi-controlled Z on all n_qubits (phase flip |11...1>)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    for q in zero_qubits:
        qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


n = N_QUBITS
num_marked = 1
optimal_iters = max(1, round((math.pi / 4) * math.sqrt(2 ** n / num_marked)))

oracle = build_oracle(n, classical_answer_int)
diffuser = build_diffuser(n)

grover = QuantumCircuit(n, n)
grover.h(range(n))
for _ in range(optimal_iters):
    grover.compose(oracle, inplace=True)
    grover.compose(diffuser, inplace=True)
grover.measure(range(n), range(n))

print(f"Grover iterations used: {optimal_iters} (optimal for 1/{2**n})")

sim = AerSimulator()
compiled = transpile(grover, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Most frequent measured bitstring (qiskit returns MSB..LSB, i.e. qubit n-1
# first, matching our marked_int formatting above).
best_bitstring = max(counts, key=counts.get)
best_count = counts[best_bitstring]
best_int = int(best_bitstring, 2)

print(f"Measurement counts (top 5): "
      f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Most frequent outcome: |{best_bitstring}> (integer {best_int}), "
      f"probability {best_count / shots:.3f}")


# ---------------------------------------------------------------------------
# 3. Compare and report.
# ---------------------------------------------------------------------------

verified = (best_int == classical_answer_int) and (best_count / shots > 0.5)

print()
print(f"Classical answer : |{classical_answer_bitstring}> ({classical_answer_int})")
print(f"Quantum result   : |{best_bitstring}> ({best_int}), "
      f"prob={best_count / shots:.3f}")

if verified:
    print("PASS")
else:
    print("FAIL")
