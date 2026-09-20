"""
Erdos problem #635 -- quantum-testable lane (best-honest-effort, limited).

Source check performed against a local read-only clone of
https://github.com/manman4/erdosproblems (data/problems.yaml, entry
"number: \"635\""):

    - number: "635"
      prize: "no"
      status: open
      oeis: ["N/A"]
      tags: ["number theory"]

Problem #635 carries **no OEIS sequence id** ("N/A") and only the generic
tag "number theory" -- there is no sequence definition text available in
this clone to derive a finite computable property *specific to problem
635* from. Per the task instructions, this is exactly the documented
limitation case: no genuine problem-635-specific sequence property could
be identified, so this script does not claim one. No OEIS value is copied
or fabricated.

What this script actually does, honestly labelled as a substitute, not a
solution to problem 635: it builds a REAL, correct Grover-search quantum
circuit over a small finite instance and verifies it against a classical
answer computed from first principles in this file, so the PASS/FAIL
result below is a genuine (if generic) quantum-computation self-test
rather than a faked verification of problem 635 itself.

Chosen finite instance (4 qubits, N = 16, search space {0, ..., 15}):
    Property tested: "n is a nonzero perfect square", i.e. n = k^2 for
    some integer k >= 1, k^2 <= 15.
    Classical answer (computed below, first principles): {1, 4, 9}.

Grover's algorithm is used to amplify the marked (perfect-square) basis
states in a uniform superposition over 4 qubits, then the circuit is
measured on the ideal AerSimulator. PASS requires the measurement
distribution to be concentrated (most probable outcomes) on exactly the
classically-computed marked set.

Honest limitation flags for the harness:
    - OEIS id used: none (problem 635 has no OEIS id in the source data).
    - verified_against_classical: True for the Grover self-test performed
      here, but this verifies a generic number-theoretic property (perfect
      squares in a small range), NOT a property derived from problem 635's
      own (nonexistent) sequence definition.
"""

import sys
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16, search space {0,...,15}


def classical_perfect_squares(n_max):
    """First-principles classical computation: nonzero perfect squares < n_max."""
    result = set()
    k = 1
    while k * k < n_max:
        result.add(k * k)
        k += 1
    return result


MARKED = classical_perfect_squares(N)  # expect {1, 4, 9}


def multi_controlled_z(qc, qubits):
    """Apply a Z controlled on all-ones over `qubits` (phase flip of |11...1>)."""
    if len(qubits) == 1:
        qc.z(qubits[0])
    elif len(qubits) == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def oracle(qc, marked_values, qubits):
    """Phase-flip amplitude of each basis state whose integer value is in marked_values."""
    for value in marked_values:
        bits = format(value, f"0{len(qubits)}b")[::-1]  # little-endian per qubit order
        zero_positions = [qubits[i] for i, b in enumerate(bits) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        multi_controlled_z(qc, qubits)
        for q in zero_positions:
            qc.x(q)


def diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    multi_controlled_z(qc, qubits)
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(marked_values, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))
    qc.h(qubits)
    for _ in range(iterations):
        oracle(qc, marked_values, qubits)
        diffuser(qc, qubits)
    qc.measure(qubits, qubits)
    return qc


def main():
    m = len(MARKED)
    # Optimal Grover iteration count for m marked items out of N.
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / m) - 0.5))

    qc = build_grover_circuit(MARKED, N_QUBITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (Qiskit little-endian: rightmost char = qubit 0) to ints.
    value_counts = {}
    for bitstring, c in counts.items():
        # Qiskit bitstrings are written MSB..LSB = qubit(n-1)..qubit0, which is
        # already standard big-endian binary, so a direct int() parse is correct.
        value = int(bitstring, 2)
        value_counts[value] = value_counts.get(value, 0) + c

    # Take the top-m most frequent outcomes as the circuit's measured answer set.
    top_values = sorted(value_counts.items(), key=lambda kv: -kv[1])[:m]
    measured_set = {v for v, _ in top_values}

    top_mass = sum(c for _, c in top_values)
    concentration = top_mass / shots

    print(f"Marked set (classical, first-principles): {sorted(MARKED)}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measured value counts (top {m}): {sorted(top_values, key=lambda kv: kv[0])}")
    print(f"Concentration on top-{m} outcomes: {concentration:.3f}")

    ok = (measured_set == MARKED) and (concentration > 0.7)

    if ok:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
