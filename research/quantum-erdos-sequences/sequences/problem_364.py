"""
Erdos problem #364 -- quantum-testable instance.

Source metadata (erdosproblems.com data, problem 364):
    oeis: ["A060355", "A076445"]
    tags: ["number theory", "powerful"]

Both OEIS sequences are built from the notion of a POWERFUL NUMBER: a
positive integer n is powerful (A001694) iff every prime p dividing n also
satisfies p^2 | n (equivalently, n = a^2 * b^3 for some positive integers
a, b). A060355 is defined as "least powerful number k such that k and k+1
are both powerful, indexed by ..." -- concretely, A060355(1) = 8: the pair
(8, 9) = (2^3, 3^2) is the smallest pair of consecutive powerful numbers.

Classical property tested here (finite, computable, and checked from first
principles in this script, not copied from OEIS):

    Over the search space n in {0, 1, ..., 63} (6 bits), find the unique n
    such that BOTH n and n+1 are powerful numbers.

We first brute-force this classically (is_powerful() below, implemented
from the prime-factorization definition) to get the ground truth: n = 8 is
the unique solution in this range, matching the known first term of
A060355 (8, with partner 9 = 8+1).

Quantum circuit: Grover's search over the 6-qubit register {0,...,63}.
The oracle is compiled directly from the classical predicate
"is_powerful(n) and is_powerful(n+1)" -- we evaluate the predicate for
every n in range, collect the (single) marked bitstring, and build a
standard multi-controlled-Z phase-flip oracle for that marked computational
basis state, sandwiched between an X-gate mask (the usual technique for
turning an arbitrary boolean predicate over a small register into a Grover
oracle when the register is small enough to enumerate). One Grover
iteration (optimal for a single marked item out of 64) is applied, and the
final measurement distribution is compared against the classical answer.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


# ---------------------------------------------------------------------
# 1. Classical ground truth (derived from first principles, no OEIS copy)
# ---------------------------------------------------------------------

def is_powerful(n: int) -> bool:
    """n > 0 is powerful iff every prime factor p of n has p^2 | n."""
    if n <= 0:
        return False
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            count = 0
            while m % p == 0:
                m //= p
                count += 1
            if count < 2:
                return False
        p += 1
    # m is 1 or a leftover prime factor with exponent 1 -> not powerful
    if m > 1:
        return False
    return True


N_QUBITS = 6
N = 2 ** N_QUBITS  # search space {0, ..., 63}

solutions = [n for n in range(N) if is_powerful(n) and n + 1 < 2 ** 20 and is_powerful(n + 1)]

assert solutions == [8], (
    f"expected the unique consecutive-powerful pair in [0,63) to be n=8, got {solutions}"
)
MARKED = solutions[0]
print(f"Classical search over n in [0,{N}): powerful(n) and powerful(n+1) -> n = {MARKED}")
print(f"  check: {MARKED} = 2^3 (powerful), {MARKED+1} = 3^2 (powerful) "
      f"-- this is OEIS A060355(1) = 8")


# ---------------------------------------------------------------------
# 2. Grover oracle for the single marked basis state
# ---------------------------------------------------------------------

def marked_bitstring(n: int, nqubits: int) -> str:
    return format(n, f"0{nqubits}b")


def build_oracle(marked: int, nqubits: int) -> QuantumCircuit:
    """Phase-flip the |marked> computational basis state."""
    qc = QuantumCircuit(nqubits, name="oracle")
    bits = marked_bitstring(marked, nqubits)[::-1]  # qubit 0 = LSB
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]

    for i in zero_positions:
        qc.x(i)

    # multi-controlled Z on all qubits: control on first n-1, target last
    qc.h(nqubits - 1)
    qc.mcx(list(range(nqubits - 1)), nqubits - 1)
    qc.h(nqubits - 1)

    for i in zero_positions:
        qc.x(i)
    return qc


def build_diffuser(nqubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(nqubits, name="diffuser")
    qc.h(range(nqubits))
    qc.x(range(nqubits))
    qc.h(nqubits - 1)
    qc.mcx(list(range(nqubits - 1)), nqubits - 1)
    qc.h(nqubits - 1)
    qc.x(range(nqubits))
    qc.h(range(nqubits))
    return qc


def grover_circuit(marked: int, nqubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(range(nqubits))
    oracle = build_oracle(marked, nqubits)
    diffuser = build_diffuser(nqubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(nqubits))
        qc.append(diffuser.to_gate(), range(nqubits))
    qc.measure(range(nqubits), range(nqubits))
    return qc


# Optimal number of Grover iterations for 1 marked item out of N=64:
# floor(pi/4 * sqrt(N/M))
iterations = int(np.floor(np.pi / 4 * np.sqrt(N / 1)))
print(f"Running Grover search with {iterations} iteration(s) over {N_QUBITS} qubits")

qc = grover_circuit(MARKED, N_QUBITS, iterations)

sim = AerSimulator()
compiled = transpile(qc, sim)
SHOTS = 4096
result = sim.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit ordering: classical register printed MSB..LSB matching qubit
# indices high..low, but we built bit 0 = qubit 0 = LSB, so reverse to read.
best_bitstring = max(counts, key=counts.get)
# Qiskit prints classical bits as c[n-1]...c[0], i.e. qubit(nqubits-1)..qubit0,
# which is exactly MSB..LSB since qubit i holds bit i of n -- so this is a
# plain binary literal, no reversal needed.
best_n = int(best_bitstring, 2)
measured_prob = counts[best_bitstring] / SHOTS

print(f"Most frequent measurement: '{best_bitstring}' -> n = {best_n} "
      f"(probability {measured_prob:.3f} over {SHOTS} shots)")

# ---------------------------------------------------------------------
# 3. Compare to classical answer
# ---------------------------------------------------------------------

ok = (best_n == MARKED) and (measured_prob > 0.5)

if ok:
    print("PASS")
else:
    print("FAIL")
