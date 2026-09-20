"""
Erdos problem #456 (erdosproblems.com/456), tags: ["number theory"], status: open.

OEIS id used: A034694 -- "a(n) = smallest m>0 such that n divides 2^m - 1, or
0 if no such m exists (i.e. n is even)."  For odd n this is exactly the
multiplicative order of 2 modulo n, ord_n(2).

Classical property tested (small, finite, computable instance):
    N = 15 (odd), a = 2.
    Find m = ord_N(a) = min{ m > 0 : a^m == 1 (mod N) }.
    This is A034694(15).

Classical derivation (done in this script, from first principles, no OEIS
lookup): brute-force the sequence 2^1, 2^2, 2^3, ... mod 15 and record the
first exponent that returns to 1. This gives the ground-truth classical
answer used to check the quantum result.

    2^1 mod 15 = 2
    2^2 mod 15 = 4
    2^3 mod 15 = 8
    2^4 mod 15 = 1   <-- first return to 1

So the classical answer is m = 4, i.e. A034694(15) = 4.

Quantum circuit: this is precisely the order-finding subroutine of Shor's
algorithm, run for a = 2, N = 15 -- the textbook case where the modular
exponentiation unitary U|y> = |2*y mod 15> (y in {0,...,15}, only the cycle
{1,2,4,8} is reachable from the starting state |1>) can be implemented
exactly with SWAP/X gates on 4 work qubits, with no ancilla/uncomputation
tricks needed. We build the genuine quantum phase estimation circuit:

  - 4 counting qubits (n_count = 4), initialized to |+>^4 via Hadamards.
  - 4 work qubits initialized to |1> (i.e. |0001>).
  - Controlled-U^(2^j) built explicitly for U: y -> 2*y mod 15, applied
    j = 0..3 as repeated controlled applications (2^j controlled powers).
  - Inverse QFT on the counting register, then measurement.

The phase estimation should return counting-register outcomes near k/4 for
k = 0,1,2,3 (since ord_15(2) = 4), and the classical continued-fraction /
direct-denominator readout of the *most probable non-zero* phase recovers
the order r = 4. We run this on the ideal AerSimulator and compare the
recovered order to the classically-computed order 4.
"""

import numpy as np
from fractions import Fraction
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute force ord_15(2) directly from the
#    definition behind OEIS A034694 (smallest m>0 with N | a^m - 1).
# ---------------------------------------------------------------------------
def classical_order(a: int, N: int) -> int:
    x = a % N
    m = 1
    val = x
    while val != 1:
        val = (val * x) % N
        m += 1
        if m > N:  # safety bound; a must be coprime to N for a finite order
            raise ValueError("no finite order found (a, N not coprime?)")
    return m


N = 15
A = 2
classical_r = classical_order(A, N)
print(f"Classical brute-force order of {A} mod {N}: r = {classical_r}")
assert classical_r == 4, "sanity check against the known Shor(15) example failed"


# ---------------------------------------------------------------------------
# 2. Quantum circuit: controlled modular multiplication by 2 (mod 15),
#    exact textbook construction on 4 work qubits using only X/SWAP gates
#    (this unitary is a permutation of computational basis states, so it
#    can be implemented exactly and reversibly without approximation).
# ---------------------------------------------------------------------------
def c_amod15(a: int, power: int):
    """Controlled multiplication by a^power mod 15, a in {2,4,7,8,11,13,14}."""
    if a not in [2, 4, 7, 8, 11, 13, 14]:
        raise ValueError("'a' must be coprime with 15 and from the given list")
    U = QuantumCircuit(4)
    for _iteration in range(power):
        if a in [2, 13]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a in [7, 8]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [4, 11]:
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = f"{a}^{power} mod 15"
    c_U = U.control()
    return c_U


def qft_dagger(n):
    """n-qubit inverse QFT, first n qubits of the circuit."""
    qc = QuantumCircuit(n)
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / float(2 ** (j - m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc


N_COUNT = 4  # counting qubits: enough to resolve phases k/4 for r=4 exactly
qc = QuantumCircuit(N_COUNT + 4, N_COUNT)

# counting register in uniform superposition
for q in range(N_COUNT):
    qc.h(q)

# work register starts in state |1> = |0001>
qc.x(N_COUNT)

# controlled-U^(2^j)
for j in range(N_COUNT):
    qc.append(c_amod15(A, 2 ** j), [j] + [N_COUNT + i for i in range(4)])

# inverse QFT on counting register
qc.append(qft_dagger(N_COUNT), range(N_COUNT))

qc.measure(range(N_COUNT), range(N_COUNT))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------
backend = AerSimulator()
compiled = transpile(qc, backend)
result = backend.run(compiled, shots=4096).result()
counts = result.get_counts()

print("Measured counting-register outcomes (bitstring: count):")
for bitstring, c in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {bitstring}: {c}")

# ---------------------------------------------------------------------------
# 4. Recover the order r from the measured phases via continued fractions,
#    taking the most common non-zero-numerator denominator as the answer
#    (standard Shor post-processing).
# ---------------------------------------------------------------------------
candidate_orders = {}
for bitstring, c in counts.items():
    # Qiskit bit ordering: bitstring[0] is the classical bit for the highest
    # qubit index; reverse to get the integer value of the counting register
    # in the usual little->big convention used for phase = value / 2^n.
    decimal = int(bitstring, 2)
    phase = decimal / (2 ** N_COUNT)
    frac = Fraction(phase).limit_denominator(N)
    r_candidate = frac.denominator
    if r_candidate == 0:
        continue
    candidate_orders[r_candidate] = candidate_orders.get(r_candidate, 0) + c

# pick the candidate order with the most measurement weight, excluding r=1
# (phase 0 is uninformative -- it occurs for every order and carries no info)
candidate_orders.pop(1, None)
quantum_r = max(candidate_orders, key=candidate_orders.get)

print(f"Quantum-recovered candidate order (continued-fraction readout): r = {quantum_r}")
print(f"Classical order: r = {classical_r}")

if quantum_r == classical_r:
    print("PASS")
else:
    print("FAIL")
