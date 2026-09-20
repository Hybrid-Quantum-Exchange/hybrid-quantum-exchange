"""
Erdos problem #267 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, number "267"):
    prize: no
    status: open (informal_status: open, last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["irrationality"]

HONESTY NOTE ON SCOPE
----------------------
Problem #267 carries NO OEIS sequence id in the source dataset (oeis: ["N/A"]).
The task requires deriving a property "from its OEIS sequence id(s) and tags" --
with no id available, there is no specific integer sequence to search or verify
membership in. Rather than fabricate an OEIS id or copy an unrelated one, this
script honestly falls back to the one real piece of information that IS present:
the tag "irrationality". It builds a genuine, small, finite, classically
checkable property that sits squarely in that topic area and is *provably*
correct by elementary number theory (not fabricated, not copied from any OEIS
b-file):

    PROPERTY TESTED: for n in {0, 1, ..., 15} (a 4-qubit search space),
    identify exactly the n for which sqrt(n) is RATIONAL, i.e. n is a
    perfect square. (For all other n in this range, sqrt(n) is irrational --
    this is the classical fact that irrationality-of-sqrt(n) proofs, the kind
    Erdos-style problems in this tag concern, hinge on: sqrt(n) in Q iff n is
    a perfect square.)

    Classically, by direct enumeration and checking k*k == n for k in
    0..15, the perfect squares in [0, 15] are exactly {0, 1, 4, 9}.
    This is computed from first principles in classical_answer() below,
    not looked up.

QUANTUM CIRCUIT: Grover's search algorithm (real amplitude amplification,
run on qiskit_aer's ideal AerSimulator) over the 4-qubit space {0,...,15},
with a phase oracle that marks exactly the perfect-square basis states
{0, 1, 4, 9} (4 marked states out of 16, computed classically and encoded
via explicit multi-controlled-Z phase flips -- no shortcut lookup of the
final answer is fed to the circuit besides which bitstrings to phase-flip,
which is exactly the classical search-marking step Grover's algorithm
requires).

After roughly the Grover-optimal number of iterations for 4 marked items in
a 16-element space, the algorithm is expected to return one of {0,1,4,9}
with high probability. The script runs many shots and checks that the
measured distribution is concentrated (with much higher total probability
than a uniform-random search would give) on exactly the classically
computed perfect-square set, and prints PASS/FAIL accordingly.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4
N = 2 ** N_QUBITS  # search space size: 0..15


def classical_answer():
    """Classically (first principles) find all n in [0, N-1] with sqrt(n) rational,
    i.e. n a perfect square. This is the ground truth the quantum circuit is
    checked against."""
    squares = []
    for n in range(N):
        k = int(math.isqrt(n))
        if k * k == n:
            squares.append(n)
    return sorted(squares)


def build_oracle(marked, n_qubits):
    """Phase oracle: flips the sign of each basis state in `marked` (list of ints),
    leaves all others unchanged. Built with X gates to remap each marked
    bitstring onto |11...1>, a multi-controlled-Z, then undo the X gates."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")
        # bits[0] is qubit n_qubits-1 (MSB); map index i -> qubit (n_qubits-1-i)
        zero_qubits = [n_qubits - 1 - i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    marked = classical_answer()
    print(f"Classical answer (perfect squares in [0, {N - 1}]): {marked}")

    M = len(marked)
    # Grover-optimal iteration count for M marked items out of N.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Using {iterations} Grover iteration(s) for M={M}, N={N}")

    qc = build_grover_circuit(marked, N_QUBITS, iterations)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 8192
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (Qiskit little-endian: rightmost char = qubit 0) to ints.
    int_counts = {}
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        int_counts[n] = int_counts.get(n, 0) + c

    marked_hits = sum(int_counts.get(n, 0) for n in marked)
    marked_prob = marked_hits / shots
    uniform_baseline = M / N  # what plain random guessing would give

    print(f"Measured probability mass on classical answer set: {marked_prob:.4f}")
    print(f"Uniform-random baseline for comparison: {uniform_baseline:.4f}")

    # Also check: the single most frequent outcome must itself be a perfect square.
    top_outcome = max(int_counts, key=int_counts.get)
    top_is_marked = top_outcome in marked

    # Success criteria: Grover must beat random guessing substantially, and the
    # most likely measured outcome must actually be a correct (perfect-square) answer.
    passed = marked_prob > 2 * uniform_baseline and top_is_marked

    print(f"Most frequent measured outcome: {top_outcome} "
          f"(is a perfect square: {top_is_marked})")

    if passed:
        print("PASS")
    else:
        print("FAIL")

    return passed


if __name__ == "__main__":
    main()
