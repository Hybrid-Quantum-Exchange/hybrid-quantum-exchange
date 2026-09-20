"""
Erdos problem #425 (erdosproblems.com), quantum-testable instance.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: 425"):
    prize: no
    status: open
    tags: ["number theory", "sidon sets"]
    oeis: ["possible"]   <-- placeholder value, not a real OEIS sequence id.

HONEST LIMITATION: problem #425's yaml entry carries no real OEIS id (the
field literally contains the string "possible", a data-entry placeholder,
not an A-number). There is therefore no OEIS sequence to test membership
in. Rather than fabricate an id or copy a literal OEIS value, this script
uses the problem's own subject matter instead: it is Erdos's Sidon-set
(B2 sequence) territory (tag "sidon sets"), the classical object behind
OEIS sequences such as A005282 (Mian-Chowla / greedy Sidon sequence).
A Sidon set (or B2 set) is a set of non-negative integers such that all
pairwise sums a+b (a <= b) are distinct.

CLASSICAL PROPERTY TESTED (small, finite, computable):
    Fix the base set S = {5, 6, 10}, which is a Sidon set (all pairwise
    sums 5+5=10, 5+6=11, 5+10=15, 6+6=12, 6+10=16, 10+10=20 are distinct).
    Search space: x in {0, 1, ..., 15} (4 qubits).
    Property: "S u {x} is still a Sidon set" (x != elements already in S,
    and adding x introduces no repeated pairwise sum). Brute-force check
    over all 16 candidates gives exactly 3 solutions, a small marked set
    well suited to Grover amplification (unlike a majority-marked set).

    This script FIRST computes, by brute-force classical enumeration over
    all 16 candidates, the exact set of x that keep S Sidon. That
    classical answer is the ground truth used to grade the quantum run.

QUANTUM CIRCUIT: a genuine Grover search circuit (AerSimulator, statevector
regime, 4 qubits for the 16 candidate values of x). The oracle is built by
applying a multi-controlled Z phase flip to each of the 4-bit basis states
that the classical computation (above) determined to be a solution -- i.e.
the marked states are derived from the real classical property, not chosen
arbitrarily. The number of Grover iterations is computed from the true
count of marked states (found classically) via the standard formula
floor(pi/4 * sqrt(N/M)). The circuit is run on AerSimulator and PASSes if
the set of x-values with the highest measured probability equals the
classical solution set.
"""

import math
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


# ---------------------------------------------------------------------------
# 1. Classical ground truth
# ---------------------------------------------------------------------------

def pairwise_sums(s):
    return {a + b for a, b in combinations(sorted(s), 2)} | {2 * a for a in s}


def is_sidon(s):
    sums = []
    for a, b in combinations(sorted(s), 2):
        sums.append(a + b)
    for a in s:
        sums.append(2 * a)
    return len(sums) == len(set(sums))


BASE = {5, 6, 10}
assert is_sidon(BASE), "base set must itself be Sidon"

N_QUBITS = 4
N = 2 ** N_QUBITS  # 16 candidate values of x

classical_solutions = sorted(
    x for x in range(N) if x not in BASE and is_sidon(BASE | {x})
)
M = len(classical_solutions)

print(f"Base Sidon set: {sorted(BASE)}")
print(f"Candidate range: x in [0, {N - 1}]")
print(f"Classical solutions (x that keep the set Sidon): {classical_solutions}")
print(f"Number of marked states M = {M} out of N = {N}")

assert 0 < M < N, "Grover needs a nontrivial (neither empty nor full) marked set"


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classical solutions
# ---------------------------------------------------------------------------

def apply_marking(qc, value, n_qubits):
    """Flip phase of |value> using a multi-controlled Z (via H + MCX + H)."""
    bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
    flipped = []
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
            flipped.append(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i in flipped:
        qc.x(i)


def oracle(n_qubits, solutions):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for v in solutions:
        apply_marking(qc, v, n_qubits)
    return qc


def diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


num_iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations: {num_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

orc = oracle(N_QUBITS, classical_solutions)
dif = diffuser(N_QUBITS)
for _ in range(num_iterations):
    qc.append(orc.to_instruction(), range(N_QUBITS))
    qc.append(dif.to_instruction(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 8192
compiled = transpile(qc, backend)
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit prints classical bits as c[n-1]...c[0], i.e. clbit i is bit i of the
# integer -- exactly the encoding apply_marking() used (bit i <-> qubit i), so
# the bitstring is read directly as a standard big-endian binary integer.
value_counts = {}
for bitstring, c in counts.items():
    x = int(bitstring, 2)
    value_counts[x] = value_counts.get(x, 0) + c

sorted_values = sorted(value_counts.items(), key=lambda kv: -kv[1])
print("Top measured candidate values (value: counts):")
for v, c in sorted_values[:max(M, 3)]:
    print(f"  x={v:2d}  count={c:5d}  {'(classical solution)' if v in classical_solutions else ''}")

top_m_quantum = sorted({v for v, _ in sorted_values[:M]})

passed = (top_m_quantum == classical_solutions)

print()
if passed:
    print(f"Quantum top-{M} candidates match classical Sidon-extension solutions: "
          f"{top_m_quantum}")
    print("PASS")
else:
    print(f"Quantum top-{M} candidates {top_m_quantum} != classical solutions "
          f"{classical_solutions}")
    print("FAIL")
