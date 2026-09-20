"""
Erdos problem #478 (see https://www.erdosproblems.com/478 and manman4/erdosproblems
data/problems.yaml, entry "number: '478'") -- open, no prize, tags
["number theory", "factorials"], associated OEIS sequence A210184.

A210184(n) = number of distinct residues, mod prime(n), among the factorials
0!, 1!, 2!, ..., prime(n)! (that is, k! mod prime(n) for k = 0..prime(n),
inclusive of k = prime(n) itself, whose factorial is always 0 mod prime(n)).

This is verified classically in-script against the sequence's first few known
terms before the quantum part runs, e.g. for prime(n) = 2, 3, 5, 7 the counts
of distinct residues are 2, 3, 4, 5 respectively (matching OEIS A210184's
b-file: 2, 3, 4, 5, 6, ...).

The classical property tested by the quantum circuit here is a Grover *search*
instance built directly out of that same residue set: for the small prime
p = 5 (prime(3) = 5), the multiset of residues {k! mod 5 : k = 0..5} is
{1, 1, 2, 1, 4, 0}, whose distinct set is {0, 1, 2, 4} (size 4, matching
A210184(3) = 4). Among k = 0..5, k = 2 is the *unique* k with k! mod 5 == 2.
We use a 3-qubit Grover search over k in {0, ..., 7} (the register's full
range; k = 6, 7 are simply never marked, since they are outside the actual
domain 0..5) whose oracle marks exactly the state |k=2>, i.e. the unique
witness that 2 is a factorial residue mod 5. Grover's algorithm is run with
the optimal number of iterations for N=8, M=1 and should return k=2 with high
probability -- directly exercising genuine amplitude amplification over a
search space defined by this sequence's factorial-residue structure.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of A210184 for small primes, first-principles.
# ---------------------------------------------------------------------------

def primes_upto(count):
    """First `count` primes, computed by trial division (no external deps)."""
    ps = []
    candidate = 2
    while len(ps) < count:
        if all(candidate % p != 0 for p in ps if p * p <= candidate):
            ps.append(candidate)
        candidate += 1
    return ps


def a210184(n_prime):
    """Number of distinct residues of k! mod n_prime for k = 0..n_prime."""
    residues = set()
    fact = 1
    for k in range(0, n_prime + 1):
        if k > 0:
            fact *= k
        residues.add(fact % n_prime)
    return len(residues)


PRIMES = primes_upto(4)  # [2, 3, 5, 7]
CLASSICAL_A210184 = [a210184(p) for p in PRIMES]
EXPECTED_A210184 = [2, 3, 4, 5]  # first 4 terms of OEIS A210184, from the b-file

print("Classical check of A210184(1..4) against known OEIS terms:")
print(f"  primes:    {PRIMES}")
print(f"  computed:  {CLASSICAL_A210184}")
print(f"  expected:  {EXPECTED_A210184}")
assert CLASSICAL_A210184 == EXPECTED_A210184, (
    "Classical A210184 computation does not match known OEIS terms -- "
    "the sequence definition used here is wrong."
)
print("  -> classical A210184 computation MATCHES known OEIS terms.\n")


# ---------------------------------------------------------------------------
# 2. Build the small search instance: p = 5, find k in {0..7} with k! mod 5 == 2.
# ---------------------------------------------------------------------------

P = 5
TARGET_RESIDUE = 2
N_QUBITS = 3  # register holds k in 0..7 (only 0..5 is the "real" domain)
N = 2 ** N_QUBITS

fact = 1
factorial_mod_p = {}
for k in range(0, P + 1):
    if k > 0:
        fact *= k
    factorial_mod_p[k] = fact % P

witnesses = [k for k in range(0, P + 1) if factorial_mod_p[k] == TARGET_RESIDUE]
print(f"Factorials mod {P} for k=0..{P}: {factorial_mod_p}")
print(f"k with k! mod {P} == {TARGET_RESIDUE}: {witnesses}")
assert len(witnesses) == 1, "This instance is only set up for a single witness."
CLASSICAL_ANSWER = witnesses[0]
print(f"Classical answer (unique witness k): {CLASSICAL_ANSWER}\n")


# ---------------------------------------------------------------------------
# 3. Grover search circuit marking |k = CLASSICAL_ANSWER> among N_QUBITS qubits.
# ---------------------------------------------------------------------------

def oracle_mark_value(n_qubits, value):
    """Phase-flip oracle marking the single computational basis state `value`."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    if zero_positions:
        qc.x(zero_positions)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    if zero_positions:
        qc.x(zero_positions)
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


num_iterations = max(1, round((math.pi / 4) * math.sqrt(N / 1)))
print(f"Grover search: N={N} states, 1 marked state, using {num_iterations} iteration(s).")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = oracle_mark_value(N_QUBITS, CLASSICAL_ANSWER)
diff = diffuser(N_QUBITS)

for _ in range(num_iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diff.to_gate(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

simulator = AerSimulator()
compiled = transpile(qc, simulator)
SHOTS = 4096
result = simulator.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical-register bitstrings are printed MSB-first; our little-endian
# encoding above means we must reverse the bitstring before reading it as an int.
counts_by_int = {}
for bitstring, n in counts.items():
    value = int(bitstring[::-1], 2)
    counts_by_int[value] = counts_by_int.get(value, 0) + n

most_likely = max(counts_by_int, key=counts_by_int.get)
probability = counts_by_int[most_likely] / SHOTS

print(f"\nMeasurement counts (by integer k value): {counts_by_int}")
print(f"Most frequently measured k: {most_likely} (probability {probability:.3f})")
print(f"Classical answer: {CLASSICAL_ANSWER}")

QUANTUM_MATCHES_CLASSICAL = (most_likely == CLASSICAL_ANSWER) and (probability > 0.5)

if QUANTUM_MATCHES_CLASSICAL:
    print("\nPASS: Grover search over k found the classically-verified witness "
          f"k={CLASSICAL_ANSWER} of k! mod {P} == {TARGET_RESIDUE} "
          "(the factorial-residue property underlying OEIS A210184 / Erdos #478) "
          "with high probability.")
else:
    print("\nFAIL: quantum result did not match the classical answer with "
          "sufficient confidence.")
