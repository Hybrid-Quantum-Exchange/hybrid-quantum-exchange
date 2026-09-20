"""
Erdos problem #691 -- quantum-testable sequence entry (best-effort / limited).

Source metadata (erdosproblems.com dataset, data/problems.yaml, entry
`number: "691"`):
    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (reported honestly, per task instructions): problem #691 carries
no OEIS sequence id in the dataset (oeis == "N/A") and no problem statement
text is available in this read-only clone beyond the tag "number theory".
There is therefore no specific integer sequence to derive a property from,
and no way to build a circuit that is genuinely *about* problem #691's
mathematical content -- doing so would require fabricating a property that
isn't actually tied to this problem, which the task explicitly forbids.

Best-effort fallback: since the only real signal available is the tag
"number theory", this script instead builds and verifies a REAL, genuine
quantum circuit for a small, classical, finite, computable number-theory
search property -- Grover's algorithm searching a 4-qubit space (N = 16,
integers 0..15) for the primes, i.e. membership in OEIS A000040 (the primes)
restricted to [0, 15]. This is a real sequence with real mathematical
content and a circuit that genuinely performs the search via a marked-state
Grover oracle, but it is NOT derived from problem #691's own statement,
since that statement is unavailable here. This limitation is real and is
not being hidden: ran_ok and verified_against_classical are reported
accurately for what was actually built and run, but this entry should not
be read as "the quantum test for Erdos problem #691's sequence" in the way
other entries in this library are, only as the best honest attempt possible
given the available metadata.

Classical property tested: for n in {0, 1, ..., 15}, is n prime?
Computed classically from first principles (trial division) below, then
used to build a Grover oracle marking exactly those basis states, and to
verify the states measured by the quantum circuit correspond to primes.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np


def apply_mcz(circ: QuantumCircuit, qubits: list[int]) -> None:
    """Multi-controlled Z on the given qubits, built from H + MCX (basis-gate friendly)."""
    target = qubits[-1]
    controls = qubits[:-1]
    circ.h(target)
    circ.mcx(controls, target)
    circ.h(target)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def build_grover_circuit(n_qubits: int, marked_states: list[int], n_iterations: int) -> QuantumCircuit:
    """Grover search over n_qubits marking the given basis states (integers)."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition
    qc.h(range(n_qubits))

    def apply_oracle(circ: QuantumCircuit) -> None:
        for state in marked_states:
            bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
            zero_positions = [i for i, b in enumerate(bits) if b == "0"]
            if zero_positions:
                circ.x(zero_positions)
            if n_qubits == 1:
                circ.z(0)
            else:
                apply_mcz(circ, list(range(n_qubits)))
            if zero_positions:
                circ.x(zero_positions)

    def apply_diffuser(circ: QuantumCircuit) -> None:
        circ.h(range(n_qubits))
        circ.x(range(n_qubits))
        if n_qubits == 1:
            circ.z(0)
        else:
            apply_mcz(circ, list(range(n_qubits)))
        circ.x(range(n_qubits))
        circ.h(range(n_qubits))

    for _ in range(n_iterations):
        apply_oracle(qc)
        apply_diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main() -> None:
    n_qubits = 4  # search space N = 16, integers 0..15
    N = 2 ** n_qubits

    # Classical answer, computed from first principles.
    primes = [n for n in range(N) if is_prime(n)]
    marked_states = primes
    M = len(marked_states)

    print(f"Search space: integers 0..{N - 1} ({n_qubits} qubits)")
    print(f"Classical property: n is prime (OEIS A000040 restricted to [0,{N - 1}])")
    print(f"Classical answer (primes in range): {primes}")

    # Optimal number of Grover iterations for M marked items out of N.
    n_iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))
    print(f"Grover iterations used: {n_iterations}")

    qc = build_grover_circuit(n_qubits, marked_states, n_iterations)

    simulator = AerSimulator()
    shots = 4096
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Aggregate measured outcomes as integers (bitstrings are big-endian in Qiskit's counts).
    outcome_counts: dict[int, int] = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        outcome_counts[value] = outcome_counts.get(value, 0) + c

    total_shots = sum(outcome_counts.values())
    marked_hits = sum(c for v, c in outcome_counts.items() if v in set(marked_states))
    marked_fraction = marked_hits / total_shots

    # Most-frequent measured outcome should itself be a genuine prime.
    most_common_value = max(outcome_counts.items(), key=lambda kv: kv[1])[0]
    most_common_is_prime = is_prime(most_common_value)

    print(f"Fraction of shots landing on a marked (prime) state: {marked_fraction:.3f}")
    print(f"Most frequently measured value: {most_common_value} (prime? {most_common_is_prime})")

    # Success criteria: Grover amplification should concentrate a clear majority
    # of shots on marked states, and the single most common outcome must be prime.
    verified = marked_fraction > 0.5 and most_common_is_prime

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
