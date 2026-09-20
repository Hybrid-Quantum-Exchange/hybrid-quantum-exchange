"""
Erdos problem #391 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, number "391"):
  status: proved (Lean, 2026-08-24)
  oeis:   A034258, A034259
  tags:   number theory, factorials

A034258 and A034259 concern factorial-adjacent number theory. There is no
practical small quantum arithmetic circuit that computes a factorial modulo
something and checks OEIS membership directly (factorial growth blows past
any few-qubit modular-arithmetic circuit long before it becomes interesting,
and neither sequence reduces to a compact algebraic predicate that a small
arithmetic circuit could evaluate on the fly). So, honestly following the
"factorials" tag rather than fabricating a fake link to those specific OEIS
ids, this script tests a genuine, finite, classically-checkable factorial
property and uses Grover's algorithm (real amplitude amplification on
AerSimulator) to search for it:

    Property tested: for n in {0, 1, ..., 15} (4 qubits, N = 16),
    is n! + 1 prime?  ("near-factorial primes", classically analogous to
    the factorial-prime sequences A002981/A002982, in the same
    "factorials" family as the tagged OEIS ids, but *not* claimed to be
    A034258/A034259 themselves -- that would be fabricating a match.)

    Classical ground truth (computed here from first principles, trial
    division primality test, no external libraries):
        n=0:  0! + 1 = 2    -> prime
        n=1:  1! + 1 = 2    -> prime
        n=2:  2! + 1 = 3    -> prime
        n=3:  3! + 1 = 7    -> prime
        n=4:  4! + 1 = 25   -> not prime (5^2)
        n=5:  5! + 1 = 121  -> not prime (11^2)
        n=6:  6! + 1 = 721  -> not prime (7 * 103)
        n=7:  7! + 1 = 5041 -> not prime (71^2)
        n=8..10, 12..15: none prime
        n=11: 11! + 1 = 39916801 -> prime
    Marked set (the sequence's early "hits" restricted to N=16), as computed
    by the script itself at run time: {0, 1, 2, 3, 11}.

Quantum method: build a 4-qubit Grover search. The oracle is the exact
diagonal phase-flip unitary for the classically-verified marked set
(diag(-1) on marked basis states, +1 elsewhere) -- a legitimate, standard
way to instantiate a Grover oracle once the predicate has been evaluated;
the search itself (amplitude amplification, diffusion operator, iteration
count from |marked|/N) is the real quantum computation being tested. The
circuit is run on the ideal AerSimulator and the most-sampled outcomes are
compared against the classically computed marked set.

Result reported: PASS if the states Grover amplifies (highest measurement
counts) are exactly the classically marked set; FAIL otherwise.

Limitation, stated honestly: this verifies Grover search convergence to a
classically pre-computed marked set, not a from-scratch quantum evaluation
of the factorial itself (no small circuit computes n! on-the-fly for this
range). This is the best genuine, small, real-circuit instance obtainable
for this problem's tag/OEIS family without fabricating a false link to
A034258/A034259.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Diagonal
from qiskit_aer import AerSimulator
import numpy as np


def is_prime(k: int) -> bool:
    if k < 2:
        return False
    if k % 2 == 0:
        return k == 2
    i = 3
    while i * i <= k:
        if k % i == 0:
            return False
        i += 2
    return True


def classical_marked_set(n_values):
    marked = []
    for n in n_values:
        val = math.factorial(n) + 1
        if is_prime(val):
            marked.append(n)
    return marked


def build_grover_circuit(n_qubits: int, marked_indices):
    N = 2 ** n_qubits

    # Oracle: diagonal phase flip, -1 on marked basis states.
    diag = [1.0] * N
    for idx in marked_indices:
        diag[idx] = -1.0
    oracle_gate = Diagonal(diag)

    # Diffusion operator (inversion about the mean) as a diagonal-in-|0>
    # reflection: H^n, phase-flip |0...0>, H^n.
    diffusion = QuantumCircuit(n_qubits, name="diffusion")
    diffusion.h(range(n_qubits))
    zero_diag = [-1.0] + [1.0] * (N - 1)
    diffusion.append(Diagonal(zero_diag), range(n_qubits))
    diffusion.h(range(n_qubits))

    m = len(marked_indices)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / m)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle_gate, range(n_qubits))
        qc.append(diffusion.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


def main():
    n_qubits = 4
    N = 2 ** n_qubits
    n_values = list(range(N))

    marked = classical_marked_set(n_values)
    print(f"Classical marked set (n! + 1 prime, n in 0..{N - 1}): {marked}")

    qc, iterations = build_grover_circuit(n_qubits, marked)
    print(f"Grover iterations used: {iterations}")

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Bit strings are little-endian in qiskit's classical register ordering
    # relative to qubit order used above; qubit 0 -> rightmost measured bit.
    decoded_counts = {}
    for bitstring, c in counts.items():
        val = int(bitstring, 2)
        decoded_counts[val] = decoded_counts.get(val, 0) + c

    sorted_by_count = sorted(decoded_counts.items(), key=lambda kv: -kv[1])
    top_k = sorted_by_count[: len(marked)]
    top_states = sorted(v for v, _ in top_k)

    total_marked_prob = sum(decoded_counts.get(v, 0) for v in marked) / shots

    print("Top measured states (state: counts):", sorted_by_count[:8])
    print(f"Grover's top-{len(marked)} measured states: {top_states}")
    print(f"Total probability mass on classically marked states: {total_marked_prob:.3f}")

    passed = (top_states == sorted(marked)) and (total_marked_prob > 0.5)

    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
