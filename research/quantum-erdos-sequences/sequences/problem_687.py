"""
Erdos problem #687 (erdosproblems.com), OEIS ids A048670 and A058989.

A048670(n) is the Jacobsthal function g(P_n) applied to the primorial
P_n = product of the first n primes: the maximal gap between consecutive
integers coprime to P_n.

A058989(n) is the closely related quantity "the largest number of
consecutive integers each of which is divisible by some prime <= the
n-th prime", i.e. the longest run of consecutive integers that are each
NOT coprime to P_n. A058989(n) = A048670(n) - 1.

Classical property tested here (n = 4, primes {2, 3, 5, 7}, P_4 = 210):
OEIS gives A058989(4) = 9, witnessed by the run 2..10 (each of 2,3,...,10
shares a factor with 210). We verify this from first principles in this
script: among the 16 candidate starting points x in [0, 15], x = 2 is
the UNIQUE value such that x, x+1, ..., x+8 (9 consecutive integers) are
all divisible by one of 2, 3, 5, 7 (i.e. gcd(x+i, 210) > 1 for every
i in 0..8). This is exactly the classical witness underlying A058989(4)=9.

Because there is a unique marked element among 16 (4 qubits), this is a
textbook Grover search instance: we build a phase oracle that flips the
sign of |x> for every x in [0,15] satisfying the "bad window" predicate
above (computed classically and hard-wired into a multi-controlled-Z
oracle), then run the standard Grover diffuser for the optimal number of
iterations (round(pi/4 * sqrt(16/1)) = 3), and measure. On the ideal
AerSimulator the most frequent measured bitstring must be x = 2 (0010),
matching the classical answer.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_QUBITS = 4          # search space size 2**4 = 16
PRIMORIAL_N = 4        # first 4 primes: 2, 3, 5, 7
PRIMES = [2, 3, 5, 7]
P_N = math.prod(PRIMES)  # 210
WINDOW = 9              # candidate run length (A058989(4))


def is_bad_window(x: int, window: int = WINDOW, modulus: int = P_N) -> bool:
    """True iff x, x+1, ..., x+window-1 are ALL non-coprime to `modulus`."""
    return all(math.gcd(x + i, modulus) > 1 for i in range(window))


def classical_search(n_qubits: int = N_QUBITS):
    """Brute-force the search space classically and return the marked set."""
    marked = [x for x in range(2 ** n_qubits) if is_bad_window(x)]
    return marked


def build_oracle(marked_values, n_qubits: int) -> QuantumCircuit:
    """Phase oracle flipping the sign of each marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked_values, n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    # 1. Classical ground truth, derived here from first principles.
    marked = classical_search(N_QUBITS)
    assert marked == [2], f"expected unique marked value [2], got {marked}"
    classical_answer = marked[0]
    # Sanity-check this really witnesses A058989(4) = 9: the run
    # classical_answer .. classical_answer + WINDOW - 1 must be exactly
    # WINDOW long and every member must share a factor with P_N.
    run = list(range(classical_answer, classical_answer + WINDOW))
    assert all(math.gcd(v, P_N) > 1 for v in run)
    assert WINDOW - 1 == 8  # A048670(4) - 1 == A058989(4), i.e. 9

    # 2. Quantum Grover search over the same 16-element space.
    n = N_QUBITS
    num_marked = len(marked)
    iterations = max(1, round((math.pi / 4) * math.sqrt((2 ** n) / num_marked)))
    qc = build_grover_circuit(marked, n, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports bitstrings as c[n-1]...c[0]; since classical bit i holds
    # qubit i (which we used as binary digit i of x), the string read left to
    # right is already the standard MSB-first binary representation of x.
    best_bitstring = max(counts, key=counts.get)
    quantum_answer = int(best_bitstring, 2)
    quantum_prob = counts[best_bitstring] / shots

    print(f"Erdos problem 687 / OEIS A048670, A058989")
    print(f"Primorial P_4 = {P_N} (primes {PRIMES}), window length = {WINDOW}")
    print(f"Classical marked value(s) among 0..15: {marked}")
    print(f"Classical answer (unique bad-window start): x = {classical_answer}")
    print(f"Grover iterations used: {iterations}")
    print(f"Quantum measurement histogram: {counts}")
    print(f"Quantum most-likely answer: x = {quantum_answer} (probability {quantum_prob:.3f})")

    verified = quantum_answer == classical_answer and quantum_prob > 0.5
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
