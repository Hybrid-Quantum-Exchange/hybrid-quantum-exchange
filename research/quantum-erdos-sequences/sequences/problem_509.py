"""
Erdos problem #509 -- quantum-testable-sequence lane (best-effort, limitation noted).

Source record checked: /home/user/manman4/erdosproblems/data/problems.yaml,
entry `number: "509"` (tags: ["analysis", "polynomials"]).

LIMITATION (read this first): problem #509's metadata lists `oeis: ["N/A"]`.
There is no OEIS sequence attached to this problem, so there is no small,
finite, computable *sequence-membership* property of "the problem 509
sequence" to build a genuine quantum test around -- one cannot be derived
honestly from the source record. Per the task's fallback instructions, this
script is a best-honest-attempt substitute rather than a fabricated
sequence property: it builds a REAL Grover search circuit that finds, among
the integers 0..63 (6 qubits), the unique n such that n is prime AND
n == 43 -- i.e. it searches a small finite space (N=64) for the marked
element 43, using a genuine phase-oracle + diffusion Grover circuit run on
AerSimulator. The "43 is prime" fact is checked classically from first
principles (trial division) in this script, and the search target is fixed
independently of that classical check, so the classical answer used for
comparison is derived, not copied from any external source.

This demonstrates the quantum-search machinery the library expects, but it
is NOT a test of Erdos problem #509 itself or of any OEIS sequence tied to
it, because no such finite computable instance exists in the source data.
Report this honestly: ran_ok can be True, but verified_against_classical
should be understood as "the Grover circuit's classical vs quantum answers
agree for this synthetic instance," not as a verification of problem 509.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_QUBITS = 6          # search space size N = 2**6 = 64
TARGET = 43            # the marked element we search for


def is_prime(n: int) -> bool:
    """Trial-division primality test, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_answer() -> int:
    """Classical ground truth: find n in [0, 64) with n prime and n == TARGET.

    This is computed independently in code (not copied from any table): we
    brute-force scan the whole space and require primality to hold for the
    fixed target, then return that unique marked element.
    """
    candidates = [n for n in range(2 ** N_QUBITS) if n == TARGET and is_prime(n)]
    assert len(candidates) == 1, "search instance must have exactly one marked element"
    return candidates[0]


def oracle_circuit(n_qubits: int, target: int) -> QuantumCircuit:
    """Phase oracle that flips the sign of the |target> basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(target, f"0{n_qubits}b")[::-1]  # little-endian bit order
    # Flip qubits that should be 0 in target, so |target> maps to |11...1>
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    # Multi-controlled Z on all qubits (phase flip when all are |1>)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    return qc


def diffuser_circuit(n_qubits: int) -> QuantumCircuit:
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


def build_grover_circuit(n_qubits: int, target: int) -> QuantumCircuit:
    n_items = 2 ** n_qubits
    # optimal number of Grover iterations for a single marked item
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_items)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = oracle_circuit(n_qubits, target)
    diffuser = diffuser_circuit(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run_grover() -> int:
    qc = build_grover_circuit(N_QUBITS, TARGET)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=2048).result()
    counts = result.get_counts()

    # bit order in Qiskit counts is big-endian string of classical bits;
    # our circuit used little-endian qubit->bit mapping via measure(range,range)
    best_bitstring = max(counts, key=counts.get)
    measured = int(best_bitstring, 2)
    return measured


def main() -> None:
    classical = classical_answer()
    quantum = run_grover()

    print(f"Erdos problem #509: no OEIS sequence attached (oeis: ['N/A']).")
    print("Best-effort substitute instance: Grover search over n in [0,64) "
          f"for n == {TARGET} and is_prime(n).")
    print(f"Classical answer: {classical}")
    print(f"Quantum (Grover) measured answer: {quantum}")

    if quantum == classical:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
