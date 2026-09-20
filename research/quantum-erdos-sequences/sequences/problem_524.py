"""
Erdos problem #524 -- quantum-testable lane (best-effort, with a noted limitation).

Source record (data/problems.yaml in the manman4/erdosproblems clone, entry for
number: "524"):
    prize: no
    informal_status: open
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["analysis", "probability", "polynomials"]

LIMITATION (read before trusting the "PASS" below as evidence about problem
524 itself): the source record gives **no OEIS sequence id** for problem 524
("N/A"). The task this script exists for asks for a small, finite, computable
property of *an OEIS sequence tied to the problem*, verified classically and
then checked with a real quantum circuit. With no sequence id there is
nothing sequence-shaped to derive that property from, so the honest options
are (a) fabricate a fake OEIS-derived claim, which is explicitly disallowed,
or (b) do a best-effort quantum computation on a small, real, checkable
mathematical object that is at least topically anchored to the problem's own
tags ("polynomials"), and say plainly that it is not a verification of
anything about problem 524's actual open conjecture (a probabilistic
statement about random polynomials, per its tags) or of any OEIS sequence.
This script takes option (b).

What it actually computes and checks
-------------------------------------
Property: for the fixed integer polynomial

    p(x) = x^3 - 2x^2 - 2x - 3

count how many integers x in the search space S = {0, 1, ..., 7} (i.e. all
3-bit unsigned integers) satisfy p(x) mod 8 == 0. This is a fully finite,
fully computable arithmetic/root-counting property in the spirit of the
"polynomials" tag -- but it is a generic small instance chosen for
demonstration, NOT a value read off any OEIS sequence (there isn't one here).

Classical ground truth is computed first, from first principles, directly in
this script (no lookup table, no hardcoded "the answer is N").

Quantum method: Grover's algorithm on 3 qubits. The oracle marks exactly the
basis states |x> for which p(x) mod 8 == 0 (built from the same classical
predicate, not from foreknowledge of which x satisfy it). With a search space
of size 8 and (as computed below) exactly 1 marked element, one Grover
iteration should amplify the marked state to near-certainty. The circuit is
run on Qiskit's ideal AerSimulator and its measured mode is compared against
the classically computed root set.

Report fields for this run: ran_ok reflects whether the script executed
without error; verified_against_classical reflects whether the quantum
circuit's measured answer matches the classical computation -- but, per the
limitation above, this is a demonstration of Grover search on a polynomial
arithmetic predicate, not a verification of Erdos problem 524 or of any of
its OEIS sequences (it has none).
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

N_QUBITS = 3
N = 2 ** N_QUBITS  # search space size = 8


def p(x: int) -> int:
    """p(x) = x^3 - 2x^2 - 2x - 3, the fixed small integer polynomial under test."""
    return x ** 3 - 2 * x ** 2 - 2 * x - 3


def classical_marked_set():
    """First-principles classical computation of {x in [0,N) : p(x) mod 8 == 0}."""
    marked = [x for x in range(N) if p(x) % 8 == 0]
    return marked


def build_oracle(marked_values, n_qubits):
    """Phase oracle flipping the sign of exactly the marked basis states."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for val in marked_values:
        bits = format(val, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            # Multi-controlled Z (phase flip on |1...1>) via H-MCX-H sandwich.
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_values, n_qubits, shots=2048):
    n_marked = len(marked_values)
    if n_marked == 0 or n_marked == 2 ** n_qubits:
        raise ValueError("Grover needs 0 < n_marked < N for a nontrivial search")

    # Optimal number of Grover iterations for this N and number of marked items.
    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt((2 ** n_qubits) / n_marked))))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    marked = classical_marked_set()
    print(f"p(x) = x^3 - 2x^2 - 2x - 3 over x in [0, {N})")
    print(f"Classical marked set (p(x) mod 8 == 0): {marked}")

    if len(marked) == 0 or len(marked) == N:
        print("Search space degenerate (0 or all marked) -- cannot run Grover. FAIL")
        return False

    counts, iterations = run_grover(marked, N_QUBITS)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    # Qiskit's classical bitstring is c[n-1]...c[0] left to right, and each
    # qubit i was measured into classical bit i, so reading the bitstring as
    # an ordinary binary number (leftmost = most significant) recovers the
    # integer value directly -- no reversal needed here (the reversal above,
    # in build_oracle, is what maps that same convention onto qubit indices).
    best_bitstring = max(counts, key=counts.get)
    measured_value = int(best_bitstring, 2)
    print(f"Most frequent measured value: {measured_value}")

    total_shots = sum(counts.values())
    marked_shots = sum(
        cnt for bs, cnt in counts.items() if int(bs, 2) in marked
    )
    marked_fraction = marked_shots / total_shots
    print(f"Fraction of shots landing on a classically-marked value: {marked_fraction:.3f}")

    passed = (measured_value in marked) and (marked_fraction > 0.9)

    print()
    if passed:
        print("PASS")
    else:
        print("FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    import sys
    sys.exit(0 if ok else 1)
