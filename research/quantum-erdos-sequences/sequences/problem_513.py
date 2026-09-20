"""
Erdos problem #513 — quantum-testable sequence attempt.

LIMITATION (read first): Problem #513's entry in erdosproblems/data/problems.yaml
has oeis: ["N/A"] and tags: ["analysis"]. There is no OEIS sequence id attached
to this problem, and its tag marks it as an analysis-flavored open problem
(informal_status: open as of 2025-08-31), not a combinatorial/number-theoretic
statement with an obvious finite, computable decision property. Erdos problems
tagged "analysis" typically concern real/complex analytic objects (growth rates,
inequalities over continuous domains, etc.) that do not reduce to a small finite
search or arithmetic predicate the way number-theoretic OEIS-backed problems do.

Per the task instructions: rather than fabricate an OEIS sequence or invent a
"property" with no real connection to problem #513, this script is an HONEST
BEST-EFFORT fallback. It does NOT test any property of problem #513 itself.
Instead it demonstrates the same class of quantum technique (Grover search)
on a small, genuinely finite, self-contained, classically-checkable number
theory decision property — "is n a member of A005843 (the even numbers) in
the search space 0..N-1" is trivial, so to keep this non-trivial and honestly
finite we instead search for prime numbers in a small range using trial
division computed from first principles in Python (no external OEIS lookup),
and verify Grover's algorithm finds exactly the marked (prime) states.

This is explicitly NOT a verification of Erdos problem #513 or of any OEIS
sequence tied to it — none exists in the source data. ran_ok / verified_against_classical
below describe only whether this fallback circuit runs correctly, not whether
problem #513 was tested.

Erdos problem number: 513
OEIS id(s) used: NONE (source data has oeis: ["N/A"] for problem 513)
Property actually tested by the circuit: primality of integers 0..7 (3 qubits),
computed classically via trial division in this script, marked with a Grover
oracle, verified against the classical set of primes in range.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMTGate, ZGate
from qiskit_aer import AerSimulator
from qiskit import transpile


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max: int):
    return sorted(n for n in range(n_max) if is_prime(n))


def build_oracle(marked_states, n_qubits):
    """Phase-flip oracle marking each state in marked_states (list of ints)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
        if zero_positions:
            qc.x(zero_positions)
    return qc


def run_grover(marked_states, n_qubits, shots=2048):
    oracle = build_oracle(marked_states, n_qubits)
    grover_op = GroverOperator(oracle)

    n_marked = len(marked_states)
    n_total = 2 ** n_qubits
    theta = np.arcsin(np.sqrt(n_marked / n_total))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(grover_op, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4
    n_max = 2 ** n_qubits  # search space 0..15

    classical_answer = classical_primes(n_max)
    print(f"Classical primes in [0, {n_max}): {classical_answer}")

    counts, iterations = run_grover(classical_answer, n_qubits, shots=4096)
    print(f"Grover iterations used: {iterations}")
    print("Measurement counts (bitstring: count):", counts)

    total_shots = sum(counts.values())
    marked_set = set(classical_answer)
    hits_on_marked = sum(
        c for bits, c in counts.items() if int(bits, 2) in marked_set
    )
    fraction_on_marked = hits_on_marked / total_shots

    print(f"Fraction of shots landing on a classically-verified prime state: "
          f"{fraction_on_marked:.4f}")

    # Grover with a properly chosen iteration count should concentrate most
    # amplitude on marked states; require a strong majority as the pass bar.
    passed = fraction_on_marked > 0.75

    print()
    print("NOTE: this circuit tests primality search via Grover's algorithm as a "
          "generic demonstration. It is NOT a test of Erdos problem #513, which "
          "has no associated OEIS id in the source data and is tagged 'analysis' "
          "(not a finite/computable combinatorial property).")
    print()
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
