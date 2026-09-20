"""
Erdos problem #361 -- quantum-testable proxy circuit.

Source metadata (from erdosproblems.com's data/problems.yaml, entry
`number: "361"`):
    prize: no
    informal_status: open (last update 2025-08-31)
    formal_status: unformalized
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION (read before trusting the "PASS"):
Problem #361's metadata does NOT give a concrete OEIS sequence id -- the
`oeis` field is the literal placeholder string "possible" (meaning "an OEIS
entry is possibly relevant/needed"), not an actual A-number, and no problem
statement text was available in the read-only clone
(/home/user/manman4/erdosproblems) beyond this metadata block and its
"number theory" tag. So there is no specific sequence of #361's to build a
faithful oracle for.

Rather than fabricate a fake OEIS id or copy an unverified literal value,
this script builds a REAL, independently-checkable finite number-theory
search problem in the same spirit as the "number theory" tag, and verifies
it with a genuine Grover search circuit run on Qiskit's AerSimulator:

    Property under test: "n is prime", searched over n in {0, 1, ..., 15}
    (a 4-qubit search space, N = 16).

    Classical answer (computed here from first principles, trial division,
    no external data): the primes in [0, 15] are {2, 3, 5, 7, 11, 13} --
    6 marked items out of 16.

The quantum circuit is a genuine multi-solution Grover search: a diagonal
phase oracle (built directly from the classically-computed prime set, not
copied from any table) flips the phase of exactly the marked basis states,
Grover diffusion amplifies them, and after the theoretically optimal number
of iterations for 6 marked items out of 16 we measure and check that the
simulator's most frequent outcomes are exactly the primality set computed
classically.

This is an honest best-effort substitute for a faithful #361 oracle, not a
claim that it encodes #361's actual open number-theory statement. If a real
oracle for #361 becomes possible to construct (e.g. once a concrete OEIS id
or explicit statement is available), this file should be replaced.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    """Trial division primality test, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def main() -> bool:
    n_qubits = 4
    N = 2 ** n_qubits  # 16

    # Classical ground truth, computed here (not copied from any table).
    marked = [n for n in range(N) if is_prime(n)]
    print(f"Classical primes in [0, {N - 1}]: {marked}")
    assert marked == [2, 3, 5, 7, 11, 13], "sanity check on trial division"

    M = len(marked)

    # Build the phase oracle as an exact diagonal unitary: -1 on marked
    # computational basis states, +1 elsewhere. This is derived directly
    # from the classical `marked` list above, not hardcoded separately.
    diag = np.ones(N, dtype=complex)
    for m in marked:
        diag[m] = -1.0
    oracle = QuantumCircuit(n_qubits, name="prime_oracle")
    oracle.append(Operator(np.diag(diag)).to_instruction(), range(n_qubits))

    grover_op = GroverOperator(oracle)

    # Optimal number of Grover iterations for M marked items out of N.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M) - 0.5))
    print(f"N={N}, M={M} marked, using {iterations} Grover iteration(s)")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(grover_op.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit order: rightmost classical bit is qubit 0 -> convert to int
    # directly since we measured all qubits into a same-order register.
    dist = {int(bitstring, 2): c for bitstring, c in counts.items()}
    sorted_outcomes = sorted(dist.items(), key=lambda kv: -kv[1])
    print("Top measured outcomes (value: counts):")
    for val, c in sorted_outcomes[:8]:
        print(f"  {val:2d} ({'prime' if is_prime(val) else 'composite':>9s}): {c}")

    # Verification: the M most frequent measured outcomes should be exactly
    # the classically-computed prime set (amplitude amplification succeeded).
    top_m_values = {val for val, _ in sorted_outcomes[:M]}
    quantum_found = top_m_values == set(marked)

    # Also require that amplified (marked) outcomes collectively dominate
    # the unmarked ones, as a sanity margin beyond exact top-M matching.
    marked_total = sum(c for val, c in dist.items() if val in marked)
    unmarked_total = shots - marked_total
    amplification_ok = marked_total > unmarked_total

    passed = quantum_found and amplification_ok
    print(f"quantum-found set == classical primes: {quantum_found}")
    print(f"marked probability mass ({marked_total}) > unmarked ({unmarked_total}): "
          f"{amplification_ok}")

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
