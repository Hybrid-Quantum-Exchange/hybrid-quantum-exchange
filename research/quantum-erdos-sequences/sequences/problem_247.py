"""
Erdos problem #247 -- quantum-testable lane.

Erdos problem #247 (see erdosproblems.com/247) is tagged "number theory,
irrationality" and its entry in data/problems.yaml lists oeis: ["N/A"] --
it has no associated OEIS sequence at all. That means the task as specified
(pick a property of *the* OEIS sequence for this problem and build a quantum
circuit that computes/verifies it) has no literal object to point at: there
is no sequence id to derive a property from.

LIMITATION (stated honestly, per instructions): this script does NOT test
any property of Erdos problem #247 itself -- there is nothing OEIS-indexed
to test. What it does instead, so the lane still contains a real, checked
quantum computation rather than nothing: it uses Grover's algorithm to
search the same kind of small, finite, computable number-theoretic space
that #247's own tags ("number theory") point at -- primality of small
integers -- and verifies the quantum search result against a classical
computation performed from first principles in this script (trial-division
primality, no external data, no OEIS lookup of any value).

Concretely:
  - Search space: n in [0, 15], encoded as 4 qubits (N = 16 = 2^4).
  - Property being tested: "n is prime" (classical primes: 2,3,5,7,11,13).
  - Classical answer: computed here by trial division, not copied from OEIS.
  - Quantum method: Grover's algorithm with an oracle built directly from
    the classical primality predicate (a phase oracle over the marked
    computational basis states), run on the ideal AerSimulator.
  - Check: after ceil(pi/4 * sqrt(N/M)) Grover iterations, sample the
    circuit and confirm the measured strings that dominate the histogram
    are exactly the classical prime set (statistical verification, as is
    standard for Grover's algorithm).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, derived from first principles (trial division).
# ---------------------------------------------------------------------------
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # search space size = 16 (n = 0..15)

classical_primes = sorted(n for n in range(N) if is_prime(n))
M = len(classical_primes)  # number of marked (prime) states
print(f"Search space: n in [0, {N - 1}] ({N_QUBITS} qubits)")
print(f"Classical primes in range (computed by trial division): {classical_primes}")
print(f"M (marked states) = {M}, N (total states) = {N}")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle: flips the phase of every basis state whose
#    integer value (little-endian qubit encoding) is prime.
# ---------------------------------------------------------------------------
def build_oracle(marked_values, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per-qubit
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        # Multi-controlled Z on all n_qubits (phase flip iff all qubits |1>)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.append(_mcz_gate(n_qubits), range(n_qubits))
        if zero_positions:
            qc.x(zero_positions)
    return qc


def _mcz_gate(n_qubits):
    from qiskit.circuit.library import ZGate
    return MCMTGate(gate=ZGate(), num_ctrl_qubits=n_qubits - 1, num_target_qubits=1)


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.append(_mcz_gate(n_qubits), range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(classical_primes, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

num_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations used: {num_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(num_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------
backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 20000
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit prints bit strings as c[n-1]...c[0] (MSB first). Our measure mapped
# qubit i -> clbit i, and qubit i carries weight 2^i (little-endian encoding
# used by the oracle/diffuser above), so clbit n-1 IS the MSB of n and
# int(bitstring, 2) recovers n directly -- no reversal needed.
value_counts = Counter()
for bitstring, count in counts.items():
    n_value = int(bitstring, 2)
    value_counts[n_value] += count

# The top-M most frequent measured values are Grover's quantum answer.
quantum_top = sorted(v for v, _ in value_counts.most_common(M))

print(f"Quantum (Grover) top-{M} measured values: {quantum_top}")
print(f"Classical primes:                        {classical_primes}")

total_marked_prob = sum(value_counts.get(v, 0) for v in classical_primes) / shots
print(f"Fraction of shots landing on a prime state: {total_marked_prob:.3f}")

verified = (quantum_top == classical_primes) and (total_marked_prob > 0.8)

if verified:
    print("PASS")
else:
    print("FAIL")
