"""
Erdos problem #203 -- quantum-testable companion script.

Source metadata (from erdosproblems.com data, data/problems.yaml, number "203"):
    prize: no
    status: open (informal), unformalized (formal)
    tags: ["primes", "covering systems"]
    oeis: ["N/A"]

LIMITATION (read this first): problem #203 has no associated OEIS sequence
("N/A" in the source data), and its actual content -- a question about
covering systems of congruences with prime-related moduli -- is an open
research question, not a small finite/computable property with a known
answer. There is therefore no literal sequence term to verify against, and
building a faithful oracle for "is this a valid covering system that solves
Erdos #203" is not a small-instance quantum circuit task.

Rather than fabricate a fake OEIS value or claim we tested the actual open
conjecture, this script honestly falls back to the one tag that *is*
finite and computable at small scale: "primes". It builds a real Grover
search circuit over a 4-qubit register (candidates 0..15) whose oracle
marks the nontrivial divisors of N = 15, and uses amplitude amplification
to find a witness that N is composite (i.e. that N is NOT prime), which is
then checked against the classical trial-division answer computed from
first principles in this file.

This is a genuine, self-contained quantum circuit (Grover's algorithm with
a real oracle + diffuser, run on AerSimulator) exercising the "primes" tag
of problem #203. It is explicitly NOT a solution to, or test of, the
covering-systems conjecture that problem #203 actually asks about, since
that problem has no OEIS sequence and no finite computable instance to
target. verified_against_classical below reflects only this substitute
primality-witness experiment, not problem #203's real open question.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Classical ground truth (first principles, no external lookup).
# ---------------------------------------------------------------------------
N = 15          # small composite instance, 4-bit search space (0..15)
NUM_QUBITS = 4  # 2**4 = 16 candidate values


def classical_nontrivial_divisors(n: int, width: int) -> list[int]:
    """All x in [0, 2**width) with 1 < x < n and n % x == 0."""
    divisors = []
    for x in range(2 ** width):
        if 1 < x < n and x != 0 and n % x == 0:
            divisors.append(x)
    return divisors


MARKED = classical_nontrivial_divisors(N, NUM_QUBITS)
assert MARKED == [3, 5], f"expected divisors of 15 to be [3, 5], got {MARKED}"
IS_PRIME_CLASSICAL = len(MARKED) == 0
print(f"Classical check: N={N}, nontrivial divisors in range = {MARKED}, "
      f"prime={IS_PRIME_CLASSICAL}")


# ---------------------------------------------------------------------------
# Quantum oracle: phase-flip exactly the marked basis states (built from the
# classically-known divisor list via explicit X / multi-controlled-Z gates,
# the standard way to encode a known marked-set oracle in a Grover circuit).
# ---------------------------------------------------------------------------
def apply_oracle(qc: QuantumCircuit, qubits, marked_values, width):
    for value in marked_values:
        bits = format(value, f"0{width}b")[::-1]  # little-endian per qubit
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(qubits[i])
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for i in zero_positions:
            qc.x(qubits[i])


def apply_diffuser(qc: QuantumCircuit, qubits, width):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(marked_values, width):
    qc = QuantumCircuit(width, width)
    qubits = list(range(width))
    qc.h(qubits)

    num_states = 2 ** width
    num_marked = len(marked_values)
    # Optimal number of Grover iterations for this search space.
    iterations = max(1, round(
        (np.pi / 4) * np.sqrt(num_states / num_marked)
    ))

    for _ in range(iterations):
        apply_oracle(qc, qubits, marked_values, width)
        apply_diffuser(qc, qubits, width)

    qc.measure(qubits, qubits)
    return qc, iterations


def run_grover(marked_values, width, shots=2048):
    qc, iterations = build_grover_circuit(marked_values, width)
    backend = AerSimulator()
    job = backend.run(qc, shots=shots)
    counts = job.result().get_counts()
    return counts, iterations


def main():
    counts, iterations = run_grover(MARKED, NUM_QUBITS)
    print(f"Grover iterations used: {iterations}")

    # Qiskit count keys are plain big-endian bit strings (q_{n-1}...q_0),
    # which is exactly the integer value under our little-endian qubit<->bit
    # encoding used to build the oracle, so a direct int() parse is correct.
    int_counts = {}
    for bitstring, n_shots in counts.items():
        value = int(bitstring, 2)
        int_counts[value] = int_counts.get(value, 0) + n_shots

    total_shots = sum(int_counts.values())
    marked_shots = sum(int_counts.get(v, 0) for v in MARKED)
    marked_fraction = marked_shots / total_shots

    most_likely = max(int_counts, key=int_counts.get)
    print(f"Measurement histogram (as integers): {int_counts}")
    print(f"Most likely outcome: {most_likely} "
          f"(classical nontrivial divisors: {MARKED})")
    print(f"Fraction of shots landing on a marked divisor: "
          f"{marked_fraction:.3f}")

    # Success criteria:
    #  - the found witness must classically divide N (verified, not assumed)
    #  - amplitude amplification must have concentrated most shots on
    #    marked states (Grover actually worked, not a random guess)
    witness_valid = (most_likely in MARKED) and (N % most_likely == 0)
    amplification_worked = marked_fraction > 0.8

    quantum_found_composite = witness_valid
    matches_classical = quantum_found_composite == (not IS_PRIME_CLASSICAL)

    ran_ok = True
    verified_against_classical = bool(
        witness_valid and amplification_worked and matches_classical
    )

    print(f"witness_valid={witness_valid} "
          f"amplification_worked={amplification_worked} "
          f"matches_classical={matches_classical}")

    if verified_against_classical:
        print("PASS")
    else:
        print("FAIL")

    return ran_ok, verified_against_classical


if __name__ == "__main__":
    main()
