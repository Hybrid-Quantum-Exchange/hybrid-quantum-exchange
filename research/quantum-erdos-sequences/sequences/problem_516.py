"""
Erdos problem #516 -- quantum-testable sequence lane.

LIMITATION (read first): as recorded in erdosproblems/data/problems.yaml,
problem #516 has oeis: ["N/A"] and tags: ["analysis"]. There is no OEIS
sequence attached to this problem, so there is no genuine integer sequence
to build a Grover/phase-estimation/amplitude-estimation oracle around, and
no small term of "the sequence" to verify against a quantum result. Any
oracle claiming to encode "the problem 516 sequence" would be fabricated,
which the task instructions explicitly forbid.

Honest fallback performed instead: since no sequence-specific property is
available, this script demonstrates the same class of technique (a Grover
search oracle over a small, exactly-classically-computable finite space)
on a property that is real, finite, and independently checked from first
principles in this file -- membership in the primes below 16, i.e. the
classical search problem "find n in {0,...,15} such that n is prime"
(the primes below 16 are OEIS A000040 truncated: 2, 3, 5, 7, 11, 13). This
is a generic, honestly-labelled placeholder, NOT a claim about problem
#516's (nonexistent) sequence. It exists so the lane produces a real,
runnable, verifiable quantum circuit rather than nothing. The search
space is 16 (not 8) so that the marked fraction (6/16 = 0.375) differs
from 1/2, since Grover amplitude amplification provably does not change
the hit probability when exactly half of the space is marked.

Classical property tested: n in {0,1,...,15} is prime.
Classical answer (computed below via trial division, not looked up):
the exact set of primes in [0,15].

Reported accurately: verified_against_classical is True only for this
placeholder property, not for any sequence belonging to problem #516,
because problem #516 has no OEIS id.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes_below_16():
    return sorted(n for n in range(16) if is_prime(n))


def build_oracle(marked, n_qubits):
    """Phase oracle flipping the sign of each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_search(marked, n_qubits, shots=2000, iterations=None):
    N = 2 ** n_qubits
    if iterations is None:
        iterations = max(1, round((np.pi / 4) * np.sqrt(N / len(marked))))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, range(n_qubits), inplace=True)
        qc.compose(diffuser, range(n_qubits), inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    n_qubits = 4  # search space {0,...,15}
    classical_answer = classical_primes_below_16()
    print(f"Classical answer (primes below 16, via trial division): {classical_answer}")

    counts = grover_search(classical_answer, n_qubits, shots=2000)
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print("Measurement counts (Qiskit bitstrings, qubit 0 = rightmost bit):", sorted_counts)

    total_shots = sum(counts.values())
    marked_hits = sum(
        c for bits, c in counts.items() if int(bits, 2) in classical_answer
    )
    marked_fraction = marked_hits / total_shots

    top_bits, _ = sorted_counts[0]
    top_value = int(top_bits, 2)

    # Theoretical hit probability for one Grover iteration on 6 marked out of
    # 16 states is sin((2*1+1)*arcsin(sqrt(6/16)))**2 ~= 0.84; require the
    # measured fraction to be comfortably above the unamplified baseline
    # (6/16 = 0.375) and close to that theoretical value.
    passed = marked_fraction > 0.75 and top_value in classical_answer

    print(f"Fraction of shots landing on a marked (prime) state: {marked_fraction:.3f}")
    print(f"Most frequent measured value: {top_value} (prime={is_prime(top_value)})")

    if passed:
        print("PASS")
    else:
        print("FAIL")

    return passed


if __name__ == "__main__":
    main()
