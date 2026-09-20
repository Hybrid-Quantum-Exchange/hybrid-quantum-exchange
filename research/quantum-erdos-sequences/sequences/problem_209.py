"""
Erdos problem #209 -- quantum-testable sequence lane.

Source metadata (from data/problems.yaml in the manman4/erdosproblems clone,
entry "number: \"209\"", record as of 2026-06-21):
    prize: no
    informal_status: disproved (Lean-formalized 2026-07-22)
    tags: ["geometry"]
    oeis: ["N/A"]

LIMITATION (read before trusting the PASS below):
Problem 209 carries no OEIS sequence id -- its "oeis" field is the literal
string "N/A". There is therefore no finite, computable *sequence* membership,
divisibility, or counting property tied to this specific problem that a
quantum circuit could test against a classical ground truth pulled from an
OEIS entry, the way the other lanes in this library do. Fabricating an OEIS
id or a fake "known term" for this problem would violate the task's explicit
instruction not to invent unfounded mathematical content, so this script does
not attempt that.

Honest fallback actually implemented below:
Since problem 209 is tagged "geometry" and offers no numeric sequence to
search over, this script instead runs a genuine, self-verifying quantum
search circuit (Grover's algorithm) on a small, well-defined, independently
computable combinatorial search problem, and is honest in its output that
this is a generic quantum-search demonstration standing in for this lane,
NOT a verification of any OEIS term for problem 209.

Chosen finite instance (unrelated to OEIS, computed here from first
principles, not copied from any table):
    Search space: all 3-bit strings x in {0,...,7}.
    Marked property: x is the unique integer in [0,7] satisfying
        x == 5   (i.e. binary "101")
    This is an arbitrary but fully specified and classically checkable
    target, chosen only to exercise a real oracle + diffusion Grover
    circuit on the ideal AerSimulator.

The script:
  1. Computes the classical answer (which 3-bit string satisfies x == 5)
     directly in Python -- trivially, but explicitly, so nothing is assumed.
  2. Builds a 3-qubit Grover oracle marking |101> and a standard diffusion
     operator, iterates the optimal number of Grover iterations for N=8,
     M=1 marked item (round(pi/4 * sqrt(N/M)) = 2 iterations).
  3. Runs it on AerSimulator, takes the most frequent measured bitstring,
     and compares it to the classical answer.
  4. Prints PASS if they match, FAIL otherwise.

verified_against_classical is honestly reported as False for the *problem
209 content itself* (there is no OEIS-derived classical fact to check here),
but the circuit's own claimed result (the state it finds) IS checked against
an independently computed classical answer for the toy search instance.
"""

import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_answer(n_qubits: int, target: int) -> str:
    """Directly compute, in Python, which n_qubits-bit string equals target.

    Returns the answer as a bitstring in Qiskit's little-endian convention
    (qubit 0 is the rightmost character), which is what circuit measurement
    results use.
    """
    if not (0 <= target < 2 ** n_qubits):
        raise ValueError("target out of range")
    # Qiskit reports classical registers with qubit 0 as the least
    # significant (rightmost) bit -- format() with binary gives exactly that
    # ordering when we read the string as-is.
    return format(target, f"0{n_qubits}b")


def build_oracle(n_qubits: int, target: int) -> QuantumCircuit:
    """Phase oracle that flips the sign of |target> and leaves all other
    computational basis states unchanged, implemented as a multi-controlled
    Z gate with X-gates flanking the qubits that must be 0 in target.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(target, f"0{n_qubits}b")[::-1]  # bits[i] is qubit i's value
    zero_qubits = [i for i, b in enumerate(bits) if b == "0"]

    for q in zero_qubits:
        qc.x(q)

    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    for q in zero_qubits:
        qc.x(q)

    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, target: int, shots: int = 4096) -> str:
    n_states = 2 ** n_qubits
    iterations = max(1, round((np.pi / 4) * np.sqrt(n_states / 1)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, target)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    transpiled = transpile(qc, sim)
    result = sim.run(transpiled, shots=shots).result()
    counts = result.get_counts()
    most_common = max(counts.items(), key=lambda kv: kv[1])[0]
    return most_common, counts, iterations


def main() -> int:
    n_qubits = 3
    target = 5  # binary 101 -- arbitrary, fully specified toy instance

    expected = classical_answer(n_qubits, target)
    measured, counts, iterations = run_grover(n_qubits, target)

    print(f"Erdos problem #209 lane -- OEIS: N/A (no sequence available)")
    print("This circuit demonstrates a generic Grover search, NOT a")
    print("verification of any problem-209-specific OEIS fact.")
    print(f"n_qubits={n_qubits}, target={target} ({expected}), "
          f"grover_iterations={iterations}")
    print(f"measurement counts: {counts}")
    print(f"classical expected bitstring: {expected}")
    print(f"quantum most-frequent bitstring: {measured}")

    ok = (measured == expected)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
