"""
Erdos problem #411 -- quantum-testable instance
=================================================

Source: erdosproblems.com problem #411 (data/problems.yaml entry, number
"411"). Metadata recorded there: prize "no", status "open" (last update
2025-08-31), OEIS id "A383044" (marked "possible" -- i.e. a candidate/likely
match rather than a confirmed formal link), tags ["number theory",
"iterated functions"], comment "partial r=2 result". The clone does not
carry the problem's full natural-language statement, and A383044's own
description was not independently available to this script at write time.

Given that, this script does NOT fabricate or copy a value that claims to
be "the" A383044 property. Instead, per the tags actually present on the
problem (number theory + iterated functions, i.e. iterating an
arithmetic function of n and asking a divisibility question about it),
it defines a genuine, small, finite, classically-computable number-theory
property in that same family, computes the classical ground truth for it
from first principles (trial-division divisor sums, no OEIS lookup, no
external data), and then uses a real Grover search circuit on
AerSimulator to find the same answer quantum-mechanically. The circuit
is a faithful unstructured-search instance; it is not evidence about
problem #411 itself, and the docstring says so plainly rather than
overstating the connection.

Classical property tested
--------------------------
For n in {1, ..., 8} (encoded as a 3-qubit basis state b = n-1, b in
0..7), define sigma(n) = sum of all positive divisors of n (the
divisor-sum function that repeated aliquot-sequence iteration, the
"iterated functions" side of the tags, is built from: aliquot(n) =
sigma(n) - n). The property is:

    P(n)  <=>  sigma(n) mod 4 == 0

sigma is computed here by direct trial division over 1..n, independently
for every n in the instance -- no table is copied from anywhere.

Classical answer for N = 8 (computed below, printed at runtime):
    sigma(1..8) = [1, 3, 4, 7, 6, 12, 8, 15]
    sigma(n) mod 4 == 0 for n in {3, 6, 7}   ->  marked basis states b in {2, 5, 6}

Quantum method
---------------
A 3-qubit Grover search (AerSimulator, statevector-exact, no noise) whose
oracle flags exactly the basis states b in {2, 5, 6} (binary 010, 101,
110), built as an explicit multi-controlled-Z phase oracle (X-gates to
remap 0-controls, multi-controlled Z, undo), followed by the standard
Grover diffuser, run for the optimal number of iterations for a
3-out-of-8 search (round(pi/4 * sqrt(8/3)) = 1 iteration). The circuit
is measured 4096 times and PASS requires the three highest-probability
outcomes to be exactly the classically-computed marked set {2, 5, 6},
together carrying a clear majority of the shots.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def sigma(n: int) -> int:
    """Sum of positive divisors of n, by direct trial division."""
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            total += d
    return total


def classical_marked_set(N: int):
    """Return sorted list of basis indices b = n-1 for n in 1..N with
    sigma(n) % 4 == 0, computed purely classically."""
    marked = []
    sigmas = []
    for n in range(1, N + 1):
        s = sigma(n)
        sigmas.append(s)
        if s % 4 == 0:
            marked.append(n - 1)
    return sorted(marked), sigmas


def build_oracle(n_qubits: int, marked_states):
    """Phase oracle flipping the sign of each marked computational basis
    state (each given as an integer 0..2**n_qubits - 1)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")
        # bits[0] is qubit n_qubits-1 ... bits[-1] is qubit 0 (MSB..LSB)
        zero_qubits = [n_qubits - 1 - i for i, c in enumerate(bits) if c == "0"]
        for q in zero_qubits:
            qc.x(q)
        # multi-controlled Z on all n_qubits (control = all but last, target = last)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, marked_states, shots: int = 4096):
    N = 2 ** n_qubits
    M = len(marked_states)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    N = 8
    n_qubits = 3

    marked, sigmas = classical_marked_set(N)
    print(f"Erdos problem #411 (OEIS A383044, tags: number theory / iterated functions)")
    print(f"Classical instance: n in 1..{N}, property P(n) <=> sigma(n) mod 4 == 0")
    print(f"sigma(1..{N}) = {sigmas}")
    print(f"Classically marked n (1-indexed): {[b + 1 for b in marked]}")
    print(f"Classically marked basis states b=n-1: {marked}")

    counts, iterations = run_grover(n_qubits, marked, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # Sort outcomes by count, descending.
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    print("Measurement counts (bitstring: count):")
    for bits, c in sorted_counts:
        print(f"  {bits}: {c}")

    top_k = sorted_counts[: len(marked)]
    top_states = sorted(int(bits, 2) for bits, _ in top_k)

    total_shots = sum(counts.values())
    marked_shots = sum(c for bits, c in counts.items() if int(bits, 2) in marked)
    marked_fraction = marked_shots / total_shots

    verified = (top_states == marked) and (marked_fraction > 0.7)

    print(f"Top-{len(marked)} measured basis states: {top_states}")
    print(f"Classically expected marked basis states: {marked}")
    print(f"Fraction of shots landing on a marked state: {marked_fraction:.3f}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
