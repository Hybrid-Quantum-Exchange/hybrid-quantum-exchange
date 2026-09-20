"""
Erdos problem #910 -- quantum-testable sequence attempt (LIMITATION CASE).

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
`number: "910"`:

    number: "910"
    prize: "no"
    informal_status: {state: "disproved", last_update: "2025-08-31"}
    status: {state: "disproved", last_update: "2025-08-31"}
    oeis: ["N/A"]
    tags: ["topology"]

There is NO OEIS sequence attached to this problem (oeis: ["N/A"]). The
problem itself is a topology statement (about covering/partitioning
structures), not a number-theoretic sequence with terms that could be
searched, counted, or tested for membership by a small quantum circuit.
Per instructions, fabricating a "sequence" or a literal OEIS-looking
property here would misrepresent the source, so this script does NOT
invent one.

Honest best-effort instead: this script builds and runs a genuine, small,
verifiable Grover-search circuit for an unrelated but well-defined finite
arithmetic property (finding the unique index i in {0,...,7} such that
i is a quadratic residue *and* prime modulo 8-space encoding is not
meaningful here either -- so to avoid any appearance of connecting a
fabricated math claim to Erdos #910, the demonstration circuit below
computes and verifies a self-contained, clearly-labeled toy fact: the
unique 3-bit integer marked by a Boolean oracle f(x) = (x == TARGET),
using Grover's algorithm on AerSimulator, and checks the quantum result
against the classical answer computed directly in this script.

This toy circuit has REAL quantum content (genuine Grover search, run on
AerSimulator, checked against a classically-computed answer) but it is
explicitly NOT derived from problem 910's mathematics, because problem
910 supplies no OEIS sequence and no finite computable numeric property
to search. That is the limitation being reported.

Classical answer (computed here, not copied): TARGET = 5 (0b101) is the
unique element of {0,...,7} satisfying x == TARGET by construction.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_answer(n_bits: int, target: int) -> int:
    """Classically find the unique x in [0, 2**n_bits) with x == target."""
    space = list(range(2 ** n_bits))
    matches = [x for x in space if x == target]
    assert len(matches) == 1, "oracle must mark exactly one element"
    return matches[0]


def build_oracle(qc: QuantumCircuit, qubits, target: int, n_bits: int):
    """Phase-flip the |target> basis state (marks it with a -1 phase)."""
    bits = format(target, f"0{n_bits}b")[::-1]  # little-endian
    for q, b in zip(qubits, bits):
        if b == "0":
            qc.x(q)
    if n_bits == 1:
        qc.z(qubits[0])
    elif n_bits == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q, b in zip(qubits, bits):
        if b == "0":
            qc.x(q)


def build_diffuser(qc: QuantumCircuit, qubits, n_bits: int):
    for q in qubits:
        qc.h(q)
        qc.x(q)
    if n_bits == 1:
        qc.z(qubits[0])
    elif n_bits == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def run_grover(n_bits: int, target: int, shots: int = 2048) -> int:
    qc = QuantumCircuit(n_bits, n_bits)
    qubits = list(range(n_bits))

    qc.h(qubits)

    n_items = 2 ** n_bits
    iterations = max(1, round((np.pi / 4) * np.sqrt(n_items)))
    for _ in range(iterations):
        build_oracle(qc, qubits, target, n_bits)
        build_diffuser(qc, qubits, n_bits)

    qc.measure(qubits, qubits)

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    most_common_bitstring = max(counts, key=counts.get)
    measured = int(most_common_bitstring, 2)
    return measured


def main():
    n_bits = 3
    target = classical_answer(n_bits, target=5)  # classically derive/check
    assert target == 5

    measured = run_grover(n_bits, target)

    print(f"Classical answer: {target}")
    print(f"Quantum (Grover) measured: {measured}")

    ok = (measured == target)
    print("PASS" if ok else "FAIL")

    print(
        "\nNOTE: Erdos problem #910 has oeis: ['N/A'] and is a topology "
        "statement with no attached OEIS sequence, so no genuine "
        "sequence-derived quantum-testable property could be constructed "
        "for it. The Grover search above is a real, correctly verified "
        "quantum circuit, but it is NOT mathematically derived from "
        "problem 910 -- it is a labeled placeholder demonstrating the "
        "limitation honestly rather than fabricating a connection."
    )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
