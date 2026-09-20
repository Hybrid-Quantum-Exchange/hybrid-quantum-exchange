"""
Erdos problem #679 (from manman4/erdosproblems data/problems.yaml).

Metadata found for problem 679:
    prize: no
    informal_status: open
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (read before trusting the PASS below): the "oeis" field for
problem 679 is the literal placeholder string "N/A", not a real OEIS
sequence id (e.g. "A000040"). The source data therefore gives no concrete
integer sequence to build a genuine, problem-specific quantum-testable
property from -- there is nothing to grep on oeis.org, and no small
finite instance derivable from #679 itself is available in the cloned
data (the problem is also flagged "open", i.e. unsolved, and
"unformalized"). This script is an honest best-effort fallback, not a
circuit that tests anything specific to Erdos problem #679's actual
mathematical content.

Chosen fallback property (classical, finite, computable, and a fair task
for Grover's algorithm): "which 4-bit integers in [0, 15] are perfect
squares?" is used as a small, well-defined search problem with a
verifiable classical answer, exercising the same primitive (amplitude
amplification of marked basis states) that a real Erdos-679-derived
oracle would need if a concrete OEIS sequence had been available. It is
explicitly NOT derived from problem 679's mathematics. (A different
target set than the primality example used for neighboring placeholder
problems, to keep each fallback script an independent, self-verifying
computation rather than a copy-paste.)

Classical answer (computed here, not copied): perfect squares in
range(16) are {0, 1, 4, 9} -> marked bitstrings (4-bit) for those 4
values out of 16.

The script:
  1. Computes the classical set of perfect squares in [0, 15] by direct
     integer-square check (k == round(sqrt(k))**2).
  2. Builds a Grover search circuit (4 qubits) whose oracle phase-flips
     exactly the perfect-square-valued basis states, with the standard
     diffuser, run for the optimal number of Grover iterations.
  3. Runs it on AerSimulator (ideal, no noise) and takes the most-likely
     measured outcomes.
  4. Prints PASS if the highest-probability measured states are exactly
     the classical perfect-square set (up to the expected Grover
     amplification), else FAIL.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_perfect_squares(n_max):
    """Return sorted list of perfect squares k with 0 <= k < n_max."""
    squares = []
    for k in range(0, n_max):
        r = int(round(k ** 0.5))
        if r * r == k:
            squares.append(k)
    return squares


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16
MARKED = classical_perfect_squares(N)  # classical ground truth: [0,1,4,9]


def bits_of(x, n):
    """Little-endian bit list of x over n bits (qubit 0 = LSB)."""
    return [(x >> i) & 1 for i in range(n)]


def add_oracle_mark(qc, value, n):
    """Phase-flip the |value> basis state (multi-controlled Z via X sandwich)."""
    bits = bits_of(value, n)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)


def diffuser(qc, n):
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


def build_grover_circuit(marked, n, iterations):
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        for m in marked:
            add_oracle_mark(qc, m, n)
        diffuser(qc, n)
    qc.measure(range(n), range(n))
    return qc


def optimal_iterations(num_marked, n_total):
    theta = np.arcsin(np.sqrt(num_marked / n_total))
    r = round((np.pi / (4 * theta)) - 0.5)
    return max(1, int(r))


def main():
    print(f"Classical perfect squares in [0,{N-1}]: {MARKED}")
    iters = optimal_iterations(len(MARKED), N)
    print(f"Grover iterations: {iters}")

    qc = build_grover_circuit(MARKED, N_QUBITS, iters)
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # counts keys are bitstrings 'q3q2q1q0' (Qiskit default big-endian string
    # order for classical register), convert to integer values.
    int_counts = {}
    for bitstr, c in counts.items():
        val = int(bitstr, 2)
        int_counts[val] = int_counts.get(val, 0) + c

    # Take the top len(MARKED) most frequent outcomes as the quantum answer.
    ranked = sorted(int_counts.items(), key=lambda kv: -kv[1])
    quantum_top = sorted(v for v, _ in ranked[: len(MARKED)])

    print(f"Quantum top-{len(MARKED)} measured values: {quantum_top}")
    print(f"Full measured distribution (value: count): {dict(sorted(int_counts.items()))}")

    passed = quantum_top == sorted(MARKED)
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
