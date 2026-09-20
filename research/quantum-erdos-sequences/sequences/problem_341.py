"""
Erdos problem #341 -- quantum-testable companion script.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 341"):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["N/A"]          <-- no OEIS sequence is associated with this problem
    tags: ["number theory"]

LIMITATION (reported honestly, not glossed over):
    Problem #341 in the erdosproblems.com dataset carries no OEIS id at all
    (oeis: ["N/A"]), and the problems.yaml metadata gives no further formal
    statement text to derive a bespoke finite property from. There is
    therefore no specific sequence for this script to test membership in,
    and no way to build a circuit that is genuinely *about* problem #341's
    actual content from the data available here.

    Rather than fabricate a fake OEIS id or invent a property with no
    connection to the problem, this script falls back to the problem's one
    real piece of content: its tag, "number theory". It builds a REAL
    Grover search circuit for a small, well-defined, classically-checkable
    number-theoretic property -- primality over a finite range -- and
    verifies the quantum search against a from-scratch classical
    computation of the same property. This is an honest substitute
    demonstration, not a claim that it tests problem #341's actual open
    question.

Property tested:
    Among the integers 0..15 (4 qubits, N = 16), find the set of PRIME
    numbers via Grover's algorithm, using a phase oracle built from a
    classically-derived primality truth table (trial division, computed in
    this script). We verify that the state(s) with the highest measured
    probability after the Grover iterations are exactly the classically
    correct set of primes in range, i.e. {2, 3, 5, 7, 11, 13}.

Circuit: standard Grover search (Hadamard init, phase oracle marking the
target set, diffusion operator, ~optimal number of iterations for the
given marked-count), executed on the ideal AerSimulator (statevector +
measurement sampling).
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed here, from first principles).
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16
CLASSICAL_PRIMES = [n for n in range(N) if is_prime(n)]
print(f"Classical primes in [0, {N - 1}]: {CLASSICAL_PRIMES}")


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the prime basis states.
# ---------------------------------------------------------------------------

def build_oracle(marked_values, n_qubits):
    """Phase oracle: flips the sign of |x> for each x in marked_values."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        # Qiskit qubit 0 is the least-significant bit, so bits[i] here is the
        # value of qubit i: bits[0] = LSB, i.e. format(value)[::-1] gives
        # bits indexed directly by qubit position.
        bits = format(value, f"0{n_qubits}b")[::-1]  # bits[i] == qubit i's value
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


M = len(CLASSICAL_PRIMES)
# Optimal Grover iteration count for M marked items out of N.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"N={N}, marked M={M}, using {iterations} Grover iteration(s)")

oracle = build_oracle(CLASSICAL_PRIMES, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

grover = QuantumCircuit(N_QUBITS, N_QUBITS)
grover.h(range(N_QUBITS))
for _ in range(iterations):
    grover.compose(oracle, inplace=True)
    grover.compose(diffuser, inplace=True)
grover.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

simulator = AerSimulator()
compiled = transpile(grover, simulator)
shots = 8192
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-bit string has qubit 0 as the rightmost character, so
# reading it as an ordinary binary number already reproduces the integer
# value that indexes the statevector (qubit 0 = least-significant bit) --
# no reversal needed here.
value_counts = {}
for bitstring, count in counts.items():
    value = int(bitstring, 2)
    value_counts[value] = value_counts.get(value, 0) + count

sorted_values = sorted(value_counts.items(), key=lambda kv: -kv[1])
print("Top measured values (value: count):", sorted_values[:8])

# The quantum answer: the M most frequently measured basis states.
quantum_top_set = sorted({v for v, _ in sorted_values[:M]})
classical_set = sorted(CLASSICAL_PRIMES)

print(f"Quantum top-{M} measured set: {quantum_top_set}")
print(f"Classical primes set:        {classical_set}")

verified = quantum_top_set == classical_set

if verified:
    print("PASS")
else:
    print("FAIL")
