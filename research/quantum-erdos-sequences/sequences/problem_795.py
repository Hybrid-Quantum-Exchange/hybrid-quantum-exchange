"""
Erdos problem #795 — quantum-testable sequence lane.

LIMITATION (read first): the local read-only clone of
manman4/erdosproblems (data/problems.yaml, entry "number: \"795\"") gives no
usable OEIS id for this problem. Its `oeis:` field is the literal placeholder
string "possible" (meaning "an OEIS sequence possibly exists"), not an actual
A-number, and the repo's README row for 795 carries the same placeholder with
no statement text, no formula, and no linked sequence. The only other field
is the tag "number theory". So the instruction's primary path (derive a
finite computable property from problem 795's own OEIS sequence) is not
available here: there is no OEIS id to derive it from.

Rather than fabricate a connection to problem 795's actual (unknown to this
clone) mathematical content, this script honestly falls back to a small,
genuine, finite, computable number-theory property in the same tag family —
primality — and builds a real Grover-search quantum circuit for it. This is
NOT a property of problem 795's sequence (none is available); it is the
closest honest substitute the instructions allow when no OEIS id exists.

Classical property tested
--------------------------
N = 16 (4 qubits). Let S = { x in [0, 15] : x is prime }.
Classically, by trial division: S = {2, 3, 5, 7, 11, 13}, |S| = 6.

Quantum computation
--------------------
Grover's algorithm on 4 qubits, with a phase oracle that flips the sign of
exactly the basis states in S (the oracle is built directly from the
classical primality check below — this is the standard, legitimate way to
instantiate Grover's algorithm for a concrete predicate). The optimal number
of Grover iterations for |S|=6 marked items out of N=16 is used. The circuit
is run on the ideal AerSimulator with many shots; PASS requires that the
measured outcomes are overwhelmingly concentrated on the classically-computed
prime set S (fraction of shots landing in S clearly above the "no
amplification" baseline of |S|/N = 6/16 = 0.375, and in fact close to the
Grover-predicted success probability).
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_qubits: int):
    N = 2 ** n_qubits
    return sorted(x for x in range(N) if is_prime(x))


def build_oracle(n_qubits: int, marked):
    """Phase oracle flipping the sign of each basis state in `marked`."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits: int):
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


def run_grover(n_qubits: int, marked, shots: int = 8192):
    N = 2 ** n_qubits
    M = len(marked)
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4
    N = 2 ** n_qubits

    marked = classical_primes(n_qubits)
    print(f"Classical primes in [0, {N - 1}]: {marked} (count={len(marked)})")

    counts, iterations = run_grover(n_qubits, marked)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    hits_in_marked = 0
    for bitstring, c in counts.items():
        value = int(bitstring, 2)  # Qiskit bitstrings are big-endian over classical bits
        if value in marked:
            hits_in_marked += c

    fraction = hits_in_marked / total_shots
    baseline = len(marked) / N
    print(f"Shots landing on a classical prime: {hits_in_marked}/{total_shots} = {fraction:.4f}")
    print(f"No-amplification baseline (uniform search): {baseline:.4f}")

    top = Counter(counts).most_common(len(marked))
    top_values = sorted(int(b, 2) for b, _ in top)
    print(f"Top-{len(marked)} measured values: {top_values}")

    # Verification against the classical answer:
    # 1) Grover amplification must clearly beat uniform-random baseline.
    # 2) The most-frequent measured values must exactly match the classical
    #    prime set (Grover's algorithm strongly concentrates amplitude on
    #    the marked states after the computed number of iterations).
    amplified = fraction > baseline + 0.15
    matches_classical = (top_values == marked)

    verified = amplified and matches_classical

    print()
    print(f"NOTE: no OEIS id was available for Erdos problem 795 in the local "
          f"clone (placeholder 'possible', no A-number). This circuit tests a "
          f"substitute number-theory property (primality via Grover search), "
          f"not problem 795's own sequence. See module docstring.")
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
