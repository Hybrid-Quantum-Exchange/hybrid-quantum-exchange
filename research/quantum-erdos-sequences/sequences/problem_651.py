"""
Erdos problem #651 -- quantum-testable lane (honest limitation notice)
========================================================================

Source record (erdosproblems.com data, data/problems.yaml, entry "number: 651"):
    prize:      no
    status:     disproved (last_update 2025-08-31)
    oeis:       ["possible"]
    tags:       ["geometry", "convex"]

"possible" in the `oeis` field is erdosproblems.com's own placeholder meaning
"an OEIS sequence may exist for this problem but none has been identified/
linked" -- it is NOT an OEIS id (OEIS ids are of the form A123456). No real
OEIS sequence id is associated with problem 651 in the source data. The
problem itself concerns a geometric/convexity statement about point
configurations, which does not reduce to a small finite integer sequence
membership/counting question the way a purely combinatorial or number-
theoretic Erdos problem would.

Per the task instructions: when no OEIS id is available (or no finite
computable property exists), the honest thing to do is note the limitation
plainly and not fabricate a property claimed to be "the" mathematical content
of problem 651. This script does exactly that: it does NOT claim to test any
property of problem 651's (nonexistent, unlinked) sequence.

What this script actually does, so the lane is not empty and still contains
a genuine, verifiable quantum computation:
  - It runs an unmodified, textbook Grover search circuit (3 qubits, N=8
    search space) on the ideal AerSimulator, marking a single classically-
    known target index via a standard multi-controlled-Z oracle.
  - It verifies the quantum measurement result against the classical answer
    (the marked index), computed directly in this script.
  - This demonstrates the quantum-search primitive that *would* be used if
    problem 651 had a linked finite sequence with a "does integer k appear"
    or "search for a witness" style question -- but it is explicitly a
    generic demonstration, not a claim about problem 651's mathematics.

Classical target (computed here, from first principles, no external data):
    N = 8 = 2**3 elements, indices 0..7.
    TARGET = 5 (an arbitrary, fixed classical choice, since there is no
    problem-651-derived target to search for).
    Grover's algorithm with optimal iteration count floor(pi/4 * sqrt(N))
    should return TARGET with high probability.

Reported outcome: ran_ok=True (circuit runs and the generic Grover search
verifies correctly against its classical answer), but
verified_against_classical should be read as "verified for the generic
demo, NOT as a verification of any property of Erdos problem 651's
sequence" -- because problem 651 has no linked OEIS sequence to test.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_answer(n_qubits: int, target: int) -> int:
    """Trivial classical computation: the target index itself, checked
    to be in range. This stands in for "the known term/answer" that a
    real OEIS-sequence-derived property would have."""
    n = 2 ** n_qubits
    if not (0 <= target < n):
        raise ValueError("target out of range")
    return target


def build_grover_circuit(n_qubits: int, target: int) -> QuantumCircuit:
    """Standard Grover search circuit marking a single computational basis
    state `target` out of 2**n_qubits, using the optimal number of Grover
    iterations."""
    n = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n)))

    qc = QuantumCircuit(n_qubits, n_qubits)

    # uniform superposition
    qc.h(range(n_qubits))

    target_bits = format(target, f"0{n_qubits}b")[::-1]  # little-endian

    def oracle(circuit: QuantumCircuit) -> None:
        # flip qubits that should be 0 in the target, so target maps to |11..1>
        for i, bit in enumerate(target_bits):
            if bit == "0":
                circuit.x(i)
        # multi-controlled Z via H-MCX-H on last qubit
        circuit.h(n_qubits - 1)
        circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        circuit.h(n_qubits - 1)
        for i, bit in enumerate(target_bits):
            if bit == "0":
                circuit.x(i)

    def diffuser(circuit: QuantumCircuit) -> None:
        circuit.h(range(n_qubits))
        circuit.x(range(n_qubits))
        circuit.h(n_qubits - 1)
        circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        circuit.h(n_qubits - 1)
        circuit.x(range(n_qubits))
        circuit.h(range(n_qubits))

    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run() -> bool:
    n_qubits = 3
    target = 5

    expected = classical_answer(n_qubits, target)

    qc = build_grover_circuit(n_qubits, target)
    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=2048).result()
    counts = result.get_counts()

    # Qiskit bit order is little-endian in the classical register string;
    # reverse to get standard integer value matching our qubit indexing.
    best_bitstring = max(counts, key=counts.get)
    measured = int(best_bitstring[::-1], 2)

    total_shots = sum(counts.values())
    target_shots = counts.get(best_bitstring, 0)
    confidence = target_shots / total_shots

    passed = (measured == expected) and (confidence > 0.5)

    print(f"Erdos problem #651: no OEIS id in source data (oeis: ['possible']).")
    print("This lane demonstrates a generic Grover search (not a test of any")
    print("property of problem 651's sequence, since none is linked).")
    print(f"N = {2**n_qubits}, classical target = {expected}")
    print(f"Grover measured (most frequent outcome) = {measured}")
    print(f"Confidence (fraction of shots on top outcome) = {confidence:.3f}")
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = run()
    if not ok:
        raise SystemExit(1)
