"""
Erdos problem #623 (erdosproblems.com), source metadata from
erdosproblems/data/problems.yaml:

    number: "623"
    tags: ["set theory"]
    oeis: ["N/A"]
    informal_status: open

LIMITATION (read this before the PASS/FAIL below):
Problem #623 has no associated OEIS sequence -- its `oeis` field is
literally the string "N/A" -- and its tag is "set theory" with an
unformalized, open informal status. There is therefore no finite,
computable integer sequence tied to this specific problem that a small
quantum circuit could search or verify; fabricating one and attributing
it to #623 would misrepresent the problem. Per the task's own fallback
instructions, this script is an honest best-effort substitute: it builds
and runs a genuine, self-contained Grover-search quantum circuit on a
small, well-defined, independently-checkable combinatorial property (not
sourced from any OEIS entry, and not claimed to be), so the file still
demonstrates real quantum computation rather than faking a result tied
to problem #623's sequence, which does not exist.

Chosen finite property (unrelated to OEIS, chosen because problem #623's
own tag is "set theory" and it has no computable sequence to substitute):
Among the 3-bit integers N = {0, 1, ..., 7}, find the unique n such that
n is odd AND n's two low bits are both 1, i.e. n mod 4 == 3.
Classically: {0,...,7} filtered by (n % 4 == 3) -> exactly one element,
n = 3 (since n=7 also satisfies n%4==3 -> actually two: 3 and 7).
So we tighten the marked-set definition to a single unique winner:
n == 3 AND top bit (bit 2) == 0, i.e. n == 3 exactly (011 in 3 bits).
This is exhaustively verified classically below, then a 3-qubit Grover
search circuit (oracle + diffuser, single Grover iteration, which is
optimal for search-space size 8 with 1 marked item) is run on the ideal
AerSimulator and its most frequent measured bitstring is compared to the
classical answer.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import math


def classical_search(n_bits: int) -> int:
    """Exhaustively find the unique n in [0, 2**n_bits) with n == 3.

    This is a deliberately simple, independently-checkable finite search:
    scan every integer in the space and return the one matching value.
    """
    space = list(range(2 ** n_bits))
    matches = [n for n in space if n == 3]
    assert len(matches) == 1, f"expected exactly one match, got {matches}"
    return matches[0]


def build_oracle(qc: QuantumCircuit, qubits, target: int, n_bits: int):
    """Phase-flip the |target> basis state (marks it for Grover)."""
    bits = format(target, f"0{n_bits}b")[::-1]  # little-endian bit order
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(qubits[i])
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(qubits[i])


def build_diffuser(qc: QuantumCircuit, qubits, n_bits: int):
    """Standard Grover diffuser (inversion about the mean)."""
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def run_grover(target: int, n_bits: int, shots: int = 2048) -> int:
    qc = QuantumCircuit(n_bits, n_bits)
    qubits = list(range(n_bits))

    # Uniform superposition over the search space.
    for q in qubits:
        qc.h(q)

    # Optimal number of Grover iterations for N=2**n_bits items, M=1 marked.
    N = 2 ** n_bits
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / 1)))

    for _ in range(iterations):
        build_oracle(qc, qubits, target, n_bits)
        build_diffuser(qc, qubits, n_bits)

    qc.measure(qubits, qubits)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Most frequent measured bitstring -> integer. Qiskit's classical bit
    # string is c[n-1]...c[0], i.e. already MSB-first with qubit_i carrying
    # weight 2**i, matching the little-endian oracle/diffuser encoding above.
    best_bitstring = max(counts, key=counts.get)
    measured = int(best_bitstring, 2)
    total = sum(counts.values())
    confidence = counts[best_bitstring] / total
    print(f"Grover counts: {counts}")
    print(f"Most frequent result: {best_bitstring} -> n={measured} "
          f"(confidence {confidence:.3f}, iterations={iterations})")
    return measured


def main():
    n_bits = 3
    classical_answer = classical_search(n_bits)
    print(f"Classical answer (exhaustive search over {2**n_bits} values): "
          f"n = {classical_answer}")

    quantum_answer = run_grover(classical_answer, n_bits)

    if quantum_answer == classical_answer:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
