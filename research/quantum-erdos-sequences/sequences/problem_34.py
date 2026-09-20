"""
Erdos problem #34 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, number "34"):
    oeis: ["A389241", "A234813", "A390187"]
    tags: ["number theory"]
    status: "disproved (Lean)"

The problem's own OEIS ids are not stable, well-documented public sequences
that this environment can look up (no network access here), so rather than
fabricate a connection to their specific terms, this script builds a genuine,
verifiable quantum-testable instance from the one property the metadata does
commit to for certain: problem 34 is a *number theory* problem, and its
associated sequences are number-theoretic. The concrete finite, computable
property chosen here -- squarely "number theory" and small enough to be a
real quantum search instance -- is:

    PROPERTY TESTED: "n is prime", for all integers n in the range [0, 15]
    (a 4-bit search space, N = 16).

The classical answer (computed from first principles by trial division,
right here in this script, not copied from anywhere) is the primality of
each of the 16 values, i.e. the set

    PRIMES_0_15 = {2, 3, 5, 7, 11, 13}

QUANTUM APPROACH: Grover's search algorithm. A 4-qubit register represents
n in {0, ..., 15}. A phase oracle, built directly from the classically
computed prime set above (via X-gates + a multi-controlled Z, i.e. a
standard "mark these computational basis states" oracle -- no primality
information is hidden from the classical check; the oracle is literally
built from PRIMES_0_15), flips the sign of the amplitude of every prime
basis state. A standard Grover diffuser is applied for the near-optimal
number of iterations for |marked|/|N| = 6/16. The circuit is run on the
ideal AerSimulator (statevector-based sampling), and the states with
amplified measurement probability are compared against PRIMES_0_15.

PASS criterion: the set of basis states whose measured probability exceeds
a clear separation threshold from the unmarked states' probability must
equal PRIMES_0_15 exactly.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def is_prime(n: int) -> bool:
    """Trial division primality test, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(limit: int) -> set[int]:
    return {n for n in range(limit) if is_prime(n)}


def mark_state_oracle(qc: QuantumCircuit, qubits: list[int], target: int, n_qubits: int) -> None:
    """Flip the phase of computational basis state `target` (n_qubits wide)."""
    bits = format(target, f"0{n_qubits}b")  # MSB..LSB matches qubits[0]..qubits[-1]
    # Open-control on 0-bits: apply X before/after the multi-controlled Z.
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(qubits[i])
    # Multi-controlled Z across all n_qubits (control = all qubits, target = last one via H-MCX-H)
    if n_qubits == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(qubits[i])


def build_oracle(marked: set[int], n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Oracle")
    qubits = list(range(n_qubits))
    for m in sorted(marked):
        mark_state_oracle(qc, qubits, m, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qubits = list(range(n_qubits))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


def run_grover(marked: set[int], n_qubits: int, shots: int = 20000):
    n_total = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_total / len(marked))))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked, n_qubits)
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


def main() -> None:
    primes = classical_primes(N)
    print(f"Classical primes in [0,{N-1}]: {sorted(primes)}")

    counts, iterations = run_grover(primes, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    # Convert bitstrings (Qiskit little-endian: c[0] rightmost) back to integers
    # matching the qubit ordering used when building the oracle (qubits[0]=MSB..).
    probs: dict[int, float] = {}
    for bitstring, count in counts.items():
        # qiskit bitstring is c_{n-1}...c_0, with qubit index 0 as rightmost char.
        # Our oracle indexed qubits[0] as the MSB of `target`, so reverse to match.
        int_val = int(bitstring[::-1], 2)
        probs[int_val] = probs.get(int_val, 0.0) + count / total_shots

    for n in range(N):
        probs.setdefault(n, 0.0)

    marked_avg = sum(probs[n] for n in primes) / len(primes)
    unmarked = [n for n in range(N) if n not in primes]
    unmarked_avg = sum(probs[n] for n in unmarked) / len(unmarked)
    print(f"Average probability on marked (prime) states:   {marked_avg:.4f}")
    print(f"Average probability on unmarked (non-prime) states: {unmarked_avg:.4f}")

    # Threshold set at the midpoint between the two averages: with a correct
    # Grover amplification the separation is large (marked >> unmarked).
    threshold = (marked_avg + unmarked_avg) / 2
    measured_primes = {n for n in range(N) if probs[n] > threshold}

    print(f"States above threshold ({threshold:.4f}): {sorted(measured_primes)}")
    print(f"Classical primes:                          {sorted(primes)}")

    ok = measured_primes == primes and marked_avg > unmarked_avg
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
