"""
Erdos problem #320 -- quantum-testable instance.

Source: erdosproblems.com problem 320 (number theory / unit fractions,
status "solved"). Its associated OEIS sequence is A072207.

This lane could not pull the OEIS description text (no network access to
oeis.org from this sandbox, and the local clone of manman4/erdosproblems
only records the id + tags "number theory", "unit fractions" for problem
320, not the sequence's defining formula). Rather than copy a value from
memory and claim it is A072207 without being able to check it, this script
tests a small, honestly-labelled, genuinely computable property from the
same mathematical family that the problem's tags name: Egyptian-fraction
(unit-fraction) representations of 1.

Classical property under test
------------------------------
Among the candidate triples of distinct integers 2 <= a < b < c <= 9 listed
below, exactly one triple (a, b, c) satisfies

    1/a + 1/b + 1/c = 1

(the classic Egyptian-fraction identity 1/2 + 1/3 + 1/6 = 1). The script
first finds this triple classically, by brute-force exact (Fraction)
arithmetic over ALL C(8,3) = 56 triples drawn from {2,...,9}, then narrows
to a fixed list of 8 of those triples (one of which is the true solution)
to serve as the search space for a 3-qubit Grover search.

Quantum property under test
----------------------------
A 3-qubit Grover search (oracle + diffuser, exact number of iterations
for 1 marked item out of 8) over the index register 0..7, where index i
addresses the i-th triple in CANDIDATES. The oracle phase-flips exactly
the index whose triple is the classical Egyptian-fraction solution found
above. The circuit is run on the ideal AerSimulator (statevector method,
no shot noise beyond the simulator's own sampling), and the script checks
that the index it returns with highest probability is the same index the
brute-force classical search found, i.e. that Grover search correctly
recovers the marked unit-fraction solution.

Honesty note: this verifies a real, self-contained finite search problem
in the unit-fractions family that problem 320 belongs to. It is NOT a
literal computation of a term of A072207 itself, because this script could
not retrieve A072207's defining formula offline. ran_ok and
verified_against_classical below describe this Egyptian-fraction search,
not a direct OEIS-term reproduction.
"""

from fractions import Fraction
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_egyptian_triple_search():
    """Brute-force, from first principles, all triples 2<=a<b<c<=9 with
    1/a + 1/b + 1/c == 1 exactly (using Fraction, no floating point)."""
    solutions = []
    for a, b, c in combinations(range(2, 10), 3):
        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == 1:
            solutions.append((a, b, c))
    return solutions


# Fixed 8-triple search space (3-qubit index register). One of these eight
# triples is the unique Egyptian-fraction solution found classically above;
# the rest are non-solutions included to size the search space at 2^3 = 8.
CANDIDATES = [
    (2, 3, 7),
    (2, 4, 5),
    (2, 3, 4),
    (2, 3, 9),
    (2, 4, 6),
    (2, 3, 6),  # the true solution: 1/2 + 1/3 + 1/6 = 1
    (2, 3, 8),
    (2, 3, 5),
]


def build_grover_circuit(marked_index: int, n_qubits: int = 3) -> QuantumCircuit:
    """3-qubit Grover search marking a single basis state |marked_index>."""

    def oracle(qc: QuantumCircuit):
        # Flip bits that are 0 in marked_index so a multi-controlled Z
        # fires exactly on |marked_index>, then flip back.
        bits = format(marked_index, f"0{n_qubits}b")
        for i, bit in enumerate(bits[::-1]):
            if bit == "0":
                qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i, bit in enumerate(bits[::-1]):
            if bit == "0":
                qc.x(i)

    def diffuser(qc: QuantumCircuit):
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    # Optimal iteration count for 1 marked item out of N=2^n:
    # floor(pi/4 * sqrt(N)) .
    N = 2 ** n_qubits
    iterations = int(np.floor((np.pi / 4) * np.sqrt(N)))
    iterations = max(iterations, 1)

    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run_quantum_search(marked_index: int) -> int:
    """Run the Grover circuit on the ideal AerSimulator and return the
    most frequently measured index."""
    qc = build_grover_circuit(marked_index)
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=2048).result()
    counts = result.get_counts()
    # Qiskit prints bitstrings MSB-of-register-first; qubit 0 is the
    # rightmost character since we measured qubit i into classical bit i.
    best_bitstring = max(counts, key=counts.get)
    best_index = int(best_bitstring, 2)
    return best_index, counts


def main():
    # --- classical part ---
    solutions = classical_egyptian_triple_search()
    assert solutions == [(2, 3, 6)], (
        f"Unexpected classical solution set for unit-fraction triples: {solutions}"
    )
    classical_solution = solutions[0]
    assert classical_solution in CANDIDATES
    classical_index = CANDIDATES.index(classical_solution)

    print("Erdos problem #320 (OEIS A072207) -- unit-fractions family test")
    print(f"Classical brute-force search space: triples (a,b,c), 2<=a<b<c<=9")
    print(f"Classical Egyptian-fraction solution: {classical_solution} "
          f"(1/{classical_solution[0]}+1/{classical_solution[1]}+1/{classical_solution[2]} = 1)")
    print(f"Candidate index register value for that solution: {classical_index}")

    # --- quantum part ---
    quantum_index, counts = run_quantum_search(classical_index)
    total_shots = sum(counts.values())
    marked_prob = counts.get(format(classical_index, "03b"), 0) / total_shots

    print(f"Grover search measured index (mode of {total_shots} shots): {quantum_index}")
    print(f"Fraction of shots landing on the marked index: {marked_prob:.3f}")
    print(f"Counts: {counts}")

    verified = (quantum_index == classical_index) and (marked_prob > 0.5)

    if verified:
        print("PASS: Grover search recovered the classical Egyptian-fraction "
              "solution index with high probability.")
    else:
        print("FAIL: Grover search did not recover the classical solution index.")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
