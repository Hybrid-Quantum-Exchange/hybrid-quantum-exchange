"""
Erdos problem #607 -- quantum-testable sequence entry.

LIMITATION (read first): problem_607 in the source data
(/home/user/manman4/erdosproblems/data/problems.yaml, entry "number: '607'")
has status "proved", tag ["geometry"], and OEIS field oeis: ["possible"] --
i.e. no actual OEIS sequence id is recorded for it (the literal string
"possible" is a placeholder in the source data, not an id). With no OEIS
id and no numeric sequence to key off of, there is no well-defined finite
computable property of "the sequence for problem 607" that this script can
honestly claim to test. Per instructions, rather than fabricate a property
or copy an invented OEIS value, this script is a best-honest-effort
substitute: a small, genuinely real, from-first-principles finite
arithmetic property, checked classically and then verified with a real
Grover-search quantum circuit on AerSimulator. It is NOT derived from
problem 607's actual mathematical content (Erdos geometry problems of this
kind, e.g. distinct-distances / unit-distance style statements, are not
reducible to a small finite oracle without a nontrivial, separate
formalization effort that is out of scope for this lane).

Chosen finite property (classical, first-principles):
  Among n in {0, 1, ..., 7} (3 qubits), find the unique n that CANNOT be
  written as a sum of two integer squares a^2 + b^2 with 0 <= a, b <= 3.
  This is a real, elementary number-theoretic property (related to the
  classical sum-of-two-squares theorem) with a small, exhaustively
  checkable search space.

The classical answer is computed in this script by brute force. A Grover
search circuit is built whose oracle marks exactly the classically-verified
non-representable value(s), run on the ideal AerSimulator, and the most
frequent measured outcome is compared against the classical answer.

Report: ran_ok reflects whether the script executed without error;
verified_against_classical reflects whether the quantum result matched the
classical computation for this generic property -- it does NOT certify any
connection to problem 607's actual (geometric) mathematical content, for
the reason stated above.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_two_square_representable(n: int, bound: int = 3) -> bool:
    """True iff n == a^2 + b^2 for some 0 <= a, b <= bound (brute force)."""
    for a in range(bound + 1):
        for b in range(bound + 1):
            if a * a + b * b == n:
                return True
    return False


def compute_classical_answer(N: int = 8, bound: int = 3):
    """Return the sorted list of n in [0, N) not representable as a sum of
    two squares with terms in [0, bound]. Computed from first principles."""
    non_representable = [n for n in range(N) if not classical_two_square_representable(n, bound)]
    return non_representable


def build_grover_circuit(marked_values, n_qubits: int, iterations: int) -> QuantumCircuit:
    """Build a Grover search circuit over n_qubits marking each value in
    marked_values (a list of ints in [0, 2**n_qubits)) with a phase flip."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition.
    qc.h(range(n_qubits))

    def apply_oracle(circuit: QuantumCircuit):
        for value in marked_values:
            bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian
            flip_qubits = [i for i, b in enumerate(bits) if b == "0"]
            if flip_qubits:
                circuit.x(flip_qubits)
            circuit.h(n_qubits - 1)
            circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            circuit.h(n_qubits - 1)
            if flip_qubits:
                circuit.x(flip_qubits)

    def apply_diffuser(circuit: QuantumCircuit):
        circuit.h(range(n_qubits))
        circuit.x(range(n_qubits))
        circuit.h(n_qubits - 1)
        circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        circuit.h(n_qubits - 1)
        circuit.x(range(n_qubits))
        circuit.h(range(n_qubits))

    for _ in range(iterations):
        apply_oracle(qc)
        apply_diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    N = 8
    n_qubits = 3
    bound = 3

    classical_answer = compute_classical_answer(N, bound)
    print(f"Classical brute-force non-representable values in [0,{N}): {classical_answer}")

    if len(classical_answer) == 0:
        print("No marked values found classically; nothing to search for. FAIL")
        return False, False

    M = len(classical_answer)
    # Optimal number of Grover iterations for M marked out of N states.
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))
    print(f"Marked count M={M}, using {iterations} Grover iteration(s)")

    qc = build_grover_circuit(classical_answer, n_qubits, iterations)

    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    shots = 4096
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (qiskit prints classical bits with qubit0 as the
    # rightmost bit, matching our little-endian encoding) to integers.
    int_counts = {}
    for bitstring, count in counts.items():
        value = int(bitstring[::-1], 2)
        int_counts[value] = int_counts.get(value, 0) + count

    print(f"Quantum measurement counts (by integer value): {int_counts}")

    marked_prob = sum(int_counts.get(v, 0) for v in classical_answer) / shots
    print(f"Total probability mass on classically-marked values: {marked_prob:.4f}")

    most_common_value = max(int_counts, key=int_counts.get)
    print(f"Most frequently measured value: {most_common_value}")

    quantum_result_correct = most_common_value in classical_answer
    high_confidence = marked_prob > 0.5

    verified = bool(quantum_result_correct and high_confidence)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return True, verified


if __name__ == "__main__":
    ran_ok = False
    verified = False
    try:
        ran_ok, verified = main()
    except Exception as exc:  # noqa: BLE001
        print(f"Script raised an exception: {exc}")
        ran_ok = False
        verified = False
