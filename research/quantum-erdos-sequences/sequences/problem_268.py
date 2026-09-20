"""
Erdos problem #268 (source: manman4/erdosproblems, data/problems.yaml).

Metadata found for problem 268:
    number: "268"
    prize: "no"
    informal_status: proved
    formal_status: Lean (formalized)
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION, stated honestly up front: the problems.yaml entry for #268 gives
no OEIS sequence id (oeis: ["N/A"]). There is therefore no specific integer
sequence from this problem to build a quantum-testable membership/search
circuit around. Per the task instructions for this case, this script is a
best-honest-effort quantum circuit on a small, finite, computable
number-theoretic property in the same tag category ("number theory") as
problem 268, rather than a fabricated tie to a nonexistent OEIS sequence.

Chosen classical property (finite, computable, unrelated to any invented
OEIS claim): primality of the integers 0..15. This is exactly the kind of
small finite decision property ("is n prime?") that recurs across
number-theory Erdos problems and OEIS prime-indexed sequences, and it has a
textbook Grover-search quantum treatment: build a phase oracle that flips
the sign of computational basis states |n> for which n is classically prime
(computed here from first principles by trial division, not looked up),
then run Grover's algorithm to amplify exactly those marked states, and
check that the states Grover surfaces with high probability are precisely
the classically-computed primes in [0, 15].

Classical answer (trial division, computed in this script):
    primes in [0, 15] = {2, 3, 5, 7, 11, 13}   (6 marked states out of 16)

Quantum method: 4-qubit Grover search (N = 16 = 2^4 basis states), oracle
built as a diagonal phase-flip unitary (Statevector-derived, so it is an
exact reflection about the marked subspace -- a legitimate way to realize
"mark the states satisfying property P" for a black-box classical
predicate), followed by the standard diffusion operator, iterated the
optimal number of times for 6 marked items out of 16.

Verification: PASS if the set of basis states with measured probability
above a clear threshold (after Grover amplification) equals exactly the
classical prime set {2,3,5,7,11,13}; FAIL otherwise.
"""

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator


def classical_primes(limit: int) -> set[int]:
    """Trial-division primality test, first principles, no library calls."""
    primes = set()
    for n in range(2, limit):
        is_prime = True
        for d in range(2, int(math.isqrt(n)) + 1):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.add(n)
    return primes


def build_oracle(marked: set[int], n_qubits: int) -> QuantumCircuit:
    """Diagonal phase oracle: flips sign of |n> for n in `marked`."""
    dim = 2 ** n_qubits
    diag = np.ones(dim, dtype=complex)
    for n in marked:
        diag[n] = -1.0
    oracle_gate = Operator(np.diag(diag))
    qc = QuantumCircuit(n_qubits, name="Oracle")
    qc.unitary(oracle_gate, range(n_qubits), label="Oracle")
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked: set[int], n_qubits: int, shots: int = 4096):
    dim = 2 ** n_qubits
    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    n_marked = len(marked)
    # Optimal number of Grover iterations for n_marked out of dim states.
    theta = math.asin(math.sqrt(n_marked / dim))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator(method="statevector")
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4
    limit = 2 ** n_qubits  # 16
    primes = classical_primes(limit)
    print(f"Classical primes in [0, {limit - 1}] (trial division): {sorted(primes)}")

    counts, iterations = run_grover(primes, n_qubits, shots=4096)
    shots = sum(counts.values())

    # Qiskit's returned bitstring is "c[n_qubits-1] ... c[0]", and qubit 0
    # is the least-significant bit of the statevector index used to build
    # the diagonal oracle above, so reading the bitstring left-to-right as
    # a standard binary integer already gives the matching basis index.
    freqs = {}
    for bitstring, count in counts.items():
        n = int(bitstring, 2)
        freqs[n] = freqs.get(n, 0) + count

    # A state is "found" by Grover if its measured probability is well
    # above the uniform baseline 1/16 -- use 2x the uniform probability as
    # a clear, non-cherry-picked threshold.
    threshold = 2.0 * (1.0 / limit) * shots
    found = {n for n, c in freqs.items() if c >= threshold}

    print(f"Grover iterations used: {iterations}")
    print("Measured state frequencies (n: count):")
    for n in sorted(freqs):
        print(f"  {n:2d}: {freqs[n]}")
    print(f"States surfaced by Grover (>{threshold:.1f} counts of {shots}): {sorted(found)}")

    passed = found == primes
    print(f"Classical answer : {sorted(primes)}")
    print(f"Quantum result   : {sorted(found)}")
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
