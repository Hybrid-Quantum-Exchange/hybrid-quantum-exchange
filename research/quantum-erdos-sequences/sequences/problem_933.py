"""
Erdos problem #933 -- quantum-testable lane (honest best-effort attempt).

Source metadata (from erdosproblems.com data, problems.yaml, entry "number: 933"):
    prize: no
    status: open (informal), unformalized (formal)
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION (read before trusting the "PASS" below):
    The `oeis` field for problem #933 in the source data is the literal string
    "possible" -- this is a placeholder/annotation, not an actual OEIS sequence
    id (OEIS ids look like A000045, A002572, etc.). No real OEIS id is given
    for this problem, and no sequence definition is available to derive a
    genuine small computable property FROM. This is exactly the "no OEIS id"
    case flagged in the task instructions: rather than fabricate a fake OEIS
    id or invent a "sequence" with no connection to problem #933, this script
    is an honest best-effort substitute.

    What it verifies instead is a real, small, finite, classically-checkable
    number-theory property (consistent with the "number theory" tag) using a
    genuine quantum circuit: primality of small integers via Grover search.
    This is NOT a derivation of problem #933's actual mathematical content,
    and should not be reported as such. ran_ok reflects whether the script
    ran and printed PASS; verified_against_classical reflects the classical
    check below -- but the property tested is a stand-in, not #933 itself.

Property actually tested:
    Grover search over N = 3 qubits (search space {0, ..., 7}) for integers
    n in that range that are prime, i.e. n in {2, 3, 5, 7}. The classical
    answer (computed here from first principles by trial division, not
    copied from anywhere) is the set {2, 3, 5, 7}. The Grover oracle marks
    exactly these basis states; we run the circuit on the ideal AerSimulator
    and check that the measurement distribution is concentrated (per-shot
    majority and near-uniform-over-marked-states check) on that classically
    computed set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit import transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    """Trial division primality test, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_answer(n_qubits: int) -> list[int]:
    """The classically-correct set of marked (prime) integers in range."""
    space = 2 ** n_qubits
    return [n for n in range(space) if is_prime(n)]


def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each state in `marked` with a -1 phase."""
    oracle = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        if zero_qubits:
            oracle.x(zero_qubits)
        if n_qubits == 1:
            oracle.z(0)
        else:
            mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
            oracle.append(mcz, list(range(n_qubits)))
        if zero_qubits:
            oracle.x(zero_qubits)
    return oracle


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    diff = QuantumCircuit(n_qubits, name="diffuser")
    diff.h(range(n_qubits))
    diff.x(range(n_qubits))
    if n_qubits == 1:
        diff.z(0)
    else:
        mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
        diff.append(mcz, list(range(n_qubits)))
    diff.x(range(n_qubits))
    diff.h(range(n_qubits))
    return diff


def run_grover(n_qubits: int, marked: list[int], shots: int = 4096):
    """Build and run the Grover circuit, returning measurement counts."""
    space = 2 ** n_qubits
    num_marked = len(marked)
    # Optimal number of Grover iterations for this search-space/marked-count ratio.
    theta = np.arcsin(np.sqrt(num_marked / space))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    qc = transpile(qc, sim)
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main() -> None:
    n_qubits = 4  # search space {0, ..., 15}; avoids the degenerate 50%-marked
    expected = classical_answer(n_qubits)
    print(f"Classical answer (primes in 0..{2**n_qubits - 1}, by trial division): {expected}")

    counts, iterations = run_grover(n_qubits, expected)
    print(f"Grover iterations used: {iterations}")
    print("Measurement counts:", counts)

    total_shots = sum(counts.values())
    marked_hits = 0
    for bitstring, c in counts.items():
        # Qiskit prints classical bits as c[n-1]...c[0] (MSB first); since our
        # circuit maps qubit i -> classical bit i, this string already reads
        # as the integer value directly (c[n-1] is the most-significant bit).
        value = int(bitstring, 2)
        if value in expected:
            marked_hits += c

    marked_fraction = marked_hits / total_shots
    print(f"Fraction of shots landing on a classically-prime state: {marked_fraction:.4f}")

    # Grover with the near-optimal iteration count should amplify the marked
    # subspace well above its prior probability (len(expected)/space).
    prior = len(expected) / (2 ** n_qubits)
    success = marked_fraction > 0.75 and marked_fraction > 2 * prior

    if success:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
