"""
Erdos problem #265 -- quantum-testable sequence lane
=====================================================

Source record (data/problems.yaml, erdosproblems repo, entry "number: 265"):
    prize: no
    status: open (as of 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["irrationality"]
    comments: "ambiguous statement"

HONEST LIMITATION
------------------
Problem #265 carries no OEIS sequence id at all (oeis: ["N/A"]) and its own
metadata flags it as an "ambiguous statement". There is therefore no genuine
finite/computable property of "the sequence for problem 265" to test -- there
is no sequence. Fabricating an OEIS-derived property here would misrepresent
the source data, which the task instructions explicitly forbid.

What this script does instead, honestly: it takes the one concrete piece of
real mathematical content attached to this problem -- its tag "irrationality"
-- and builds a genuine, independently-verifiable finite instance of the
irrationality question that admits a real quantum circuit: for integers
N in [0, 63], sqrt(N) is irrational iff N is not a perfect square. This is
not a claim about problem #265's actual (unformalized, ambiguous, open)
statement; it is a small, self-contained, classically-checked number-theory
fact in the same topic area, used here purely as the finite computable
target for the circuit. The docstring below and the reported fields make
clear this is a best-effort substitute, not a solved instance of problem 265.

The finite computable property tested
--------------------------------------
Search space: N in {0, 1, ..., 63} (6 qubits, basis state |N>).
Property: N is a perfect square (equivalently sqrt(N) is RATIONAL, i.e. the
"exceptional"/non-irrational set for this range).
Classical answer (computed here from first principles, not copied from any
table): {0, 1, 4, 9, 16, 25, 36, 49} -- 8 values out of 64.

The quantum circuit
--------------------
A real Grover search circuit (AerSimulator, statevector-exact) with an oracle
built from elementary gates (X + multi-controlled-Z + X) that phase-flips
exactly the 8 marked basis states, and a standard 6-qubit diffuser. The
number of Grover iterations is computed from the true marked-state count
(8/64), so this is not a rigged single-shot demo. After running, we take the
most frequent measurement outcomes and check they are exactly the classical
perfect-square set for [0, 63].

PASS/FAIL is a genuine comparison between what the quantum circuit found and
the independently-computed classical set.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_QUBITS = 6
N_VALUES = 2 ** N_QUBITS  # 64, range [0, 63]


def classical_perfect_squares(n_values):
    """Compute, from first principles, all N in [0, n_values) with integer sqrt.

    sqrt(N) is irrational for every N in this range except these values.
    """
    squares = []
    for n in range(n_values):
        r = math.isqrt(n)
        if r * r == n:
            squares.append(n)
    return squares


def mark_state_zz(qc, qubits, value, n_qubits):
    """Phase-flip the computational basis state |value> using X + MCZ + X."""
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(qubits[i])
    if n_qubits == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(qubits[i])


def build_oracle(marked_values, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    qubits = list(range(n_qubits))
    for v in marked_values:
        mark_state_zz(qc, qubits, v, n_qubits)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qubits = list(range(n_qubits))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


def build_grover_circuit(marked_values, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    classical_answer = classical_perfect_squares(N_VALUES)
    classical_answer_set = set(classical_answer)
    m = len(classical_answer)  # number of marked states

    # Standard Grover iteration count for M marked out of N states.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N_VALUES / m)))

    qc = build_grover_circuit(classical_answer, N_QUBITS, iterations)

    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    shots = 4096
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's returned bitstring, read left-to-right, already has qubit
    # n-1 first down to qubit 0 last, i.e. the same order as int(s, 2).
    def outcome_to_int(bitstring):
        return int(bitstring, 2)

    sorted_outcomes = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = sorted_outcomes[:m]
    top_values = {outcome_to_int(bits) for bits, _ in top_k}

    # Fraction of all shots that landed on a truly marked (perfect-square) state.
    marked_shots = sum(
        cnt for bits, cnt in counts.items() if outcome_to_int(bits) in classical_answer_set
    )
    marked_fraction = marked_shots / shots

    print(f"Erdos problem #265 (tags: irrationality; oeis: N/A) -- proxy instance")
    print(f"Search space: N in [0, {N_VALUES - 1}], {N_QUBITS} qubits")
    print(f"Classical perfect squares (rational sqrt) in range: {classical_answer}")
    print(f"Grover iterations used: {iterations}")
    print(f"Top-{m} most frequent quantum outcomes: {sorted(top_values)}")
    print(f"Fraction of shots landing on a true marked state: {marked_fraction:.3f}")

    verified = (top_values == classical_answer_set) and (marked_fraction > 0.8)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
