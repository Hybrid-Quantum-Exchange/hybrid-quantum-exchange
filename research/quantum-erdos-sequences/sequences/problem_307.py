"""
Erdos problem #307 (erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: \"307\"", tags: ["number theory", "unit fractions"]).

LIMITATION, stated up front: this entry's `oeis` field is `["N/A"]` — the
source repository does not attach an OEIS sequence id to problem 307. There
is therefore no OEIS sequence to build a quantum-testable circuit "from" in
the way other lanes in this library do. What follows is the best honest
substitute: a genuine, finite, computable number-theory search drawn
directly from the problem's own tags ("unit fractions"), verified
classically from first principles in this script, and then solved with a
real Grover search circuit on Qiskit's AerSimulator. It is not a stand-in
for an OEIS-anchored result, and this script's ran_ok / verified_against_classical
should be read as "the unit-fraction search below checks out", not as
"problem 307 has been reduced to a quantum circuit".

The property tested
--------------------
A classic unit-fraction (Egyptian fraction) identity:

    1/2 + 1/3 + 1/c = 1

has exactly one positive-integer solution for c. We treat "find c" as a
search over a small finite space c in {0, 1, ..., 7} (3 qubits), and ask
a Grover oracle to mark the unique c satisfying the identity exactly
(checked with Python's `fractions.Fraction`, no floating point).

Classical ground truth (computed here, not copied from anywhere):
    1/2 + 1/3 = 5/6, so 1/c must equal 1/6, i.e. c = 6.
Among c in {0,...,7}, c=0 is skipped (division by zero / not a unit
fraction), and brute force over c in {1,...,7} confirms c=6 is the unique
solution. That gives a single marked basis state out of 8, which is
exactly Grover's canonical use case, run here on a real (simulated)
quantum circuit rather than assumed.

Circuit
-------
3 qubits index c in {0,...,7} via computational basis states |c>.
A phase oracle flips the sign of |6> (binary 110) using X gates on the
qubits that must be 0 plus a multi-controlled Z, then Grover diffusion is
applied floor(pi/4 * sqrt(N)) ~= 2 times for N=8. The circuit is run on
AerSimulator (ideal, no noise) and the most-sampled outcome is compared
against the classically-computed answer c=6.
"""

from fractions import Fraction

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, derived from first principles (no OEIS lookup).
# ---------------------------------------------------------------------------

def classical_search(bound: int = 8) -> int:
    """Brute-force the unique c in {1,...,bound-1} with 1/2+1/3+1/c == 1."""
    target = Fraction(1, 1)
    partial = Fraction(1, 2) + Fraction(1, 3)
    solutions = []
    for c in range(1, bound):
        if partial + Fraction(1, c) == target:
            solutions.append(c)
    if len(solutions) != 1:
        raise RuntimeError(f"expected exactly one solution, found {solutions}")
    return solutions[0]


CLASSICAL_ANSWER = classical_search(bound=8)
assert CLASSICAL_ANSWER == 6, "sanity check on the hand-derived identity failed"


# ---------------------------------------------------------------------------
# 2. Real Grover search circuit marking |CLASSICAL_ANSWER> among 3 qubits.
# ---------------------------------------------------------------------------

N_QUBITS = 3
N = 2 ** N_QUBITS  # search space size = 8


def build_oracle(marked: int, n_qubits: int) -> QuantumCircuit:
    """Phase oracle that flips the sign of |marked> (little-endian bits)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked, f"0{n_qubits}b")[::-1]  # qubit 0 = LSB
    zero_qubits = [i for i, b in enumerate(bits) if b == "0"]

    for q in zero_qubits:
        qc.x(q)

    # Multi-controlled Z on all n_qubits (control on n_qubits-1, target phase
    # flip realised via H-MCX-H on the last qubit).
    if n_qubits == 1:
        qc.z(0)
    else:
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

    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked: int, n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def optimal_grover_iterations(n: int) -> int:
    iterations = int(np.floor((np.pi / 4) * np.sqrt(n)))
    return max(iterations, 1)


def run_grover(marked: int, n_qubits: int, shots: int = 2048):
    iterations = optimal_grover_iterations(2 ** n_qubits)
    circuit = build_grover_circuit(marked, n_qubits, iterations)

    backend = AerSimulator()
    transpiled = transpile(circuit, backend)
    job = backend.run(transpiled, shots=shots)
    counts = job.result().get_counts()

    # Bitstrings from Qiskit are big-endian in the printed key (qubit n-1 first,
    # qubit 0 last), matching the register order used in the measure call.
    most_likely_bits = max(counts, key=counts.get)
    most_likely_value = int(most_likely_bits, 2)
    return most_likely_value, counts, iterations


def main():
    print(f"Erdos problem #307 (unit fractions) -- classical answer: c = {CLASSICAL_ANSWER}")
    print(f"Grover search space: {N} basis states over {N_QUBITS} qubits")

    quantum_answer, counts, iterations = run_grover(CLASSICAL_ANSWER, N_QUBITS)

    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")
    print(f"Most frequent measured value: {quantum_answer}")
    print(f"Classical answer:            {CLASSICAL_ANSWER}")

    total_shots = sum(counts.values())
    marked_bits = format(CLASSICAL_ANSWER, f"0{N_QUBITS}b")
    marked_prob = counts.get(marked_bits, 0) / total_shots
    print(f"Probability mass on marked state |{marked_bits}>: {marked_prob:.3f}")

    verified = (quantum_answer == CLASSICAL_ANSWER) and (marked_prob > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
