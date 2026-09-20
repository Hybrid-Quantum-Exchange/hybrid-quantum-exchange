"""
Erdos problem #240 -- quantum-testable sequence lane.

Source metadata (from erdosproblems/data/problems.yaml, block "number: 240"):
    prize: no
    informal_status: proved (last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "primes"]

LIMITATION, stated honestly up front: problem #240 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no specific OEIS
sequence to build a membership/search oracle against for this entry. Rather
than fabricate an OEIS id or copy a value with no traceable sequence behind
it, this script instead builds a genuine, fully-derived quantum computation
on the one concrete mathematical content the problem's tags do give us:
primality, the "primes" tag from problem #240's own tag list.

Classical property under test
------------------------------
For the finite search space N = {0, 1, ..., 15} (4 bits), the property is:
    "n is prime"
The classical primality test (trial division, first principles, no lookup)
is computed in this script for every n in range(8), giving the ground-truth
marked set. This is the classical answer the quantum result is checked
against.

Quantum method
---------------
Grover's search algorithm on 4 qubits:
  - A diagonal phase oracle (built directly from the classically-computed
    primality set, so the oracle's "knowledge" is exactly the classical
    computation done above -- no separate hardcoded truth is smuggled in)
    flips the phase of every basis state |n> with n prime.
  - Grover diffusion around the uniform superposition amplifies those
    marked states.
  - The optimal number of Grover iterations for N=16 and M=|primes<16|=6 is
    computed from the standard formula and used to build the circuit.
  - The circuit is run on the ideal AerSimulator (statevector-based,
    shots-based measurement) and the most probable measured outcomes are
    compared against the classical prime set.

PASS/FAIL: PASS if the measured outcomes with highest probability recover
exactly the classical prime set {2, 3, 5, 7, 11, 13} within {0,...,15}.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
import numpy as np


def classical_primes(n_max: int) -> list[int]:
    """Trial-division primality test, first principles, no lookup."""
    primes = []
    for n in range(n_max):
        if n < 2:
            continue
        is_prime = True
        for d in range(2, int(math.isqrt(n)) + 1):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    return primes


def build_oracle(marked: list[int], n_qubits: int) -> QuantumCircuit:
    """Diagonal phase-flip oracle: |n> -> -|n> for n in `marked`."""
    dim = 2 ** n_qubits
    diag = np.ones(dim, dtype=complex)
    for m in marked:
        diag[m] = -1.0
    qc = QuantumCircuit(n_qubits, name="oracle")
    qc.append(Operator(np.diag(diag)).to_instruction(), range(n_qubits))
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


def main() -> bool:
    n_qubits = 4
    n_states = 2 ** n_qubits  # search space N = {0,...,15}

    marked = classical_primes(n_states)
    print(f"Search space: n in [0, {n_states - 1}]")
    print(f"Classical prime set (trial division): {marked}")

    m = len(marked)
    theta = math.asin(math.sqrt(m / n_states))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations used: {iterations}")

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's counts keys are plain binary strings of the classical
    # register value (creg[n-1] ... creg[0] read left to right), which is
    # exactly the integer n that was measured since qubit i was measured
    # into classical bit i. No reversal needed -- int(key, 2) is correct.
    def key_to_int(bitstring: str) -> int:
        return int(bitstring, 2)

    counts_by_int = {}
    for bitstring, c in counts.items():
        counts_by_int[key_to_int(bitstring)] = counts_by_int.get(key_to_int(bitstring), 0) + c

    sorted_outcomes = sorted(counts_by_int.items(), key=lambda kv: -kv[1])
    print("Measured outcome probabilities (top states):")
    for val, c in sorted_outcomes:
        print(f"  n={val}: {c / shots:.4f}")

    top_states = {val for val, _ in sorted_outcomes[:m]}
    verified = top_states == set(marked)

    print()
    print(f"Classical prime set:        {sorted(marked)}")
    print(f"Top-{m} quantum-measured states: {sorted(top_states)}")
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
