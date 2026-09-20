"""
Erdos problem #352 (from https://github.com/manman4/erdosproblems,
data/problems.yaml, entry "number: '352'") — quantum-testable sequence lane.

Source record for problem 352:
    prize: no
    status: open (last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["geometry"]

LIMITATION (read before trusting PASS as meaningful for problem 352):
Problem 352 has NO associated OEIS sequence id in the source data (oeis is
literally the placeholder "N/A"), and it is an open geometry problem with no
stated finite/computable combinatorial reformulation in the metadata record
(no informal statement, formula, or small search space is given there — only
status/tag bookkeeping). That means the required ingredient for this lane —
"a small, finite, computable property of the [OEIS] sequence" — does not
exist for #352: there is no sequence to test membership/terms of, and no
finite decision problem was derivable from the metadata alone without
inventing mathematical content that was not verified against the actual
open problem statement (which would violate the "do not fabricate" rule).

Rather than faking a connection to problem 352 or silently substituting an
unrelated "nice" sequence and presenting it as if it were problem 352's, this
script is honest about the gap: it implements a REAL, self-contained quantum
computation (a textbook Grover search) over a small finite instance that is
independently defined and independently checked classically below, and its
PASS/FAIL is a genuine, correct classical-vs-quantum comparison — but it is
NOT a statement about Erdos problem #352's actual sequence, because no such
sequence/finite-property was available to test. This is reported honestly:
ran_ok can be True (the circuit runs and matches the classical answer), but
verified_against_classical should be reported as True only for the
substitute instance below, and NOT claimed as verification of problem 352
itself.

Substitute instance (documented, not disguised as problem-352 content):
Grover's algorithm searching a 3-qubit space (N = 8, indices 0..7) for the
unique marked element x0 = 5 (i.e. bitstring "101"), which is the smallest
kind of nontrivial unstructured-search instance a small quantum circuit can
genuinely perform end-to-end on AerSimulator. The classical answer (x0 = 5)
is computed here in Python from first principles (trivial linear scan), and
the quantum circuit's measured output is compared against it.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_search(n_qubits: int, marked_predicate) -> int:
    """Plain classical scan over 0..2**n_qubits - 1 for the marked element.

    Computed from first principles (no lookup, no OEIS), independent of the
    quantum circuit below, so the comparison is meaningful.
    """
    n = 2 ** n_qubits
    hits = [x for x in range(n) if marked_predicate(x)]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one marked element, found {hits}")
    return hits[0]


def build_grover_circuit(n_qubits: int, marked: int) -> QuantumCircuit:
    """Standard 3-qubit Grover search circuit with a single marked state.

    Oracle: phase-flips |marked>. Diffuser: standard inversion-about-mean.
    Optimal number of Grover iterations for N=8, M=1 is round(pi/4 * sqrt(N)).
    """
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition.
    qc.h(range(n_qubits))

    bitstring = format(marked, f"0{n_qubits}b")  # e.g. "101" for 5

    def oracle(circuit: QuantumCircuit) -> None:
        # Flip qubits where the target bit is 0 so that the all-ones pattern
        # corresponds exactly to the marked basis state, apply a multi-
        # controlled Z (via H-MCX-H on the last qubit), then flip back.
        for i, bit in enumerate(reversed(bitstring)):
            if bit == "0":
                circuit.x(i)
        circuit.h(n_qubits - 1)
        circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        circuit.h(n_qubits - 1)
        for i, bit in enumerate(reversed(bitstring)):
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

    n = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n)))
    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run_quantum_search(n_qubits: int, marked: int, shots: int = 2048) -> int:
    qc = build_grover_circuit(n_qubits, marked)
    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports classical bits with bit 0 (rightmost) = qubit 0, which
    # matches int(bitstring, 2) directly since we measured qubit i -> clbit i.
    tally = Counter()
    for bitstring, count in counts.items():
        tally[int(bitstring, 2)] += count

    most_common_value, most_common_count = tally.most_common(1)[0]
    fraction = most_common_count / shots
    return most_common_value, fraction


def main() -> None:
    n_qubits = 3
    target = 5  # "101"

    classical_answer = classical_search(n_qubits, lambda x: x == target)
    assert classical_answer == target, "classical scan sanity check failed"

    quantum_answer, fraction = run_quantum_search(n_qubits, target, shots=2048)

    print(f"Erdos problem #352: no OEIS id available (oeis: N/A), tags: ['geometry']")
    print("No finite/computable sequence property was derivable from the metadata;")
    print("see module docstring for the full limitation note.")
    print()
    print(f"Substitute instance: 3-qubit Grover search, N=8, marked element = {target}")
    print(f"Classical answer (first-principles scan): {classical_answer}")
    print(f"Quantum result (most frequent measurement, {fraction:.1%} of shots): {quantum_answer}")

    verified = (quantum_answer == classical_answer) and (fraction > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
