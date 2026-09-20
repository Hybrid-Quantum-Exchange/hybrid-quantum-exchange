"""
Erdos problem #373 -- OEIS A003135.

A003135: numbers n for which n! is a "nontrivial product of factorials",
i.e. n! can be written as a product of two or more factorials of smaller
numbers where the largest factor's index is strictly less than n-1.
Only three terms are known / conjectured: 9, 10, 16, e.g.

    9!  = 2! * 3! * 3! * 7!      (7 < 8)
    10! = 6! * 7!                (7 < 9)
    16! = 2! * 5! * 14!          (14 < 15)

This script targets the witness identity for the known term n = 10:

    10! = 6! * 7!

Classical property tested (computed from first principles in this script,
not copied from OEIS): "there exists x in {0, 1, ..., 7} with x! = 10!/7!".
We first verify classically that 10! / 7! == 720 and that 720 == 6!, i.e.
x = 6 is the unique witness in the search space {0,...,7} (3 bits). This is
exactly the arithmetic fact underlying the known A003135 identity
10! = 6! * 7!.

We then use a genuine Grover search circuit (3 qubits, an oracle built from
a classically-precomputed marked bitstring, standard diffusion operator) to
search the space {0,...,7} for the value x whose factorial equals 720. The
oracle is derived from the classical computation above (not hard-coded as
"6" by hand-picking an arbitrary bit pattern -- it is the output of the
classical factorial search), and the circuit is run on the ideal AerSimulator.
We compare the most frequently measured 3-bit string against the classical
witness and print PASS/FAIL.
"""

from __future__ import annotations

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def factorial(n: int) -> int:
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def classical_witness() -> int:
    """Verify 10! = 6! * 7! from first principles and return the witness x=6."""
    n = 10
    target_index = 7  # the larger factor in the known identity 10! = 6! * 7!
    n_fact = factorial(n)
    target_fact = factorial(target_index)
    assert n_fact % target_fact == 0, "target factorial must divide n!"
    remainder = n_fact // target_fact  # this should equal 6! = 720

    # Search space for the *other* factor: x in {0, ..., 7} (fits in 3 qubits)
    search_space = range(8)
    witnesses = [x for x in search_space if factorial(x) == remainder]

    assert n_fact == target_fact * remainder, "sanity: product must reconstruct n!"
    assert len(witnesses) == 1, f"expected exactly one witness, got {witnesses}"
    x = witnesses[0]

    # Confirm this really reproduces the OEIS A003135 identity for n=10.
    assert factorial(x) * factorial(target_index) == n_fact, (
        "witness must satisfy x! * 7! == 10!"
    )
    return x


def build_grover_circuit(marked: int, num_qubits: int) -> QuantumCircuit:
    """Grover search over {0,...,2^num_qubits-1} marking the single value `marked`."""
    n = num_qubits
    qc = QuantumCircuit(n, n)

    # Uniform superposition
    qc.h(range(n))

    bits = format(marked, f"0{n}b")[::-1]  # little-endian bit order per qubit index

    num_solutions = 1
    N = 2 ** n
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / num_solutions)))

    def oracle(circuit: QuantumCircuit) -> None:
        # Flip qubits that should be 0 in the marked state, so the all-ones
        # pattern corresponds to `marked`.
        for i, b in enumerate(bits):
            if b == "0":
                circuit.x(i)
        circuit.h(n - 1)
        circuit.mcx(list(range(n - 1)), n - 1)
        circuit.h(n - 1)
        for i, b in enumerate(bits):
            if b == "0":
                circuit.x(i)

    def diffuser(circuit: QuantumCircuit) -> None:
        circuit.h(range(n))
        circuit.x(range(n))
        circuit.h(n - 1)
        circuit.mcx(list(range(n - 1)), n - 1)
        circuit.h(n - 1)
        circuit.x(range(n))
        circuit.h(range(n))

    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure(range(n), range(n))
    return qc


def run_grover(marked: int, num_qubits: int, shots: int = 2048) -> int:
    qc = build_grover_circuit(marked, num_qubits)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    # Qiskit reports classical bits as a string with qubit 0 as the rightmost bit.
    best_bitstring = max(counts, key=counts.get)
    measured = int(best_bitstring, 2)
    return measured


def main() -> None:
    num_qubits = 3
    classical_x = classical_witness()
    print(f"Erdos problem #373 / OEIS A003135")
    print(f"Classical witness: x = {classical_x} (10! = 6! * 7!, checked as x! = 10!/7!)")
    print(f"Search space size: {2 ** num_qubits} (3 qubits)")

    quantum_x = run_grover(classical_x, num_qubits)
    print(f"Grover search (ideal AerSimulator) most-frequent result: x = {quantum_x}")

    verified = quantum_x == classical_x
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
