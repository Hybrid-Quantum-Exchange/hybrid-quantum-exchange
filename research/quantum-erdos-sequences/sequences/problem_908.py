"""
Erdos problem #908 -- quantum-testable sequence attempt.

LIMITATION (read first): problem_908's entry in erdosproblems/data/problems.yaml
has oeis: ["N/A"] and tags: ["analysis"]. There is no OEIS sequence attached to
this problem, so there is no concrete integer sequence to build a genuine
quantum search/oracle circuit around for THIS problem. Rather than fabricate a
property and pretend it comes from problem 908's actual mathematical content
(which is an analytic statement, not a finite combinatorial/number-theoretic
one), this script is honest about that gap.

To still deliver a real, verifiable quantum computation (per the task's
fallback instruction), this script performs Grover's algorithm to search a
small, genuinely computable, well-defined property: finding the primes among
{0, ..., 15} (i.e. membership in OEIS A000040, the primes), a finite decidable
property with real mathematical content, computed classically from first
principles (trial division) inside this script and then verified by running
an actual Grover search circuit on Qiskit's AerSimulator whose marked states
are exactly those classical primes.

This is NOT a claim that A000040 is problem 908's OEIS sequence -- it is a
stand-in used only because problem 908 supplies no OEIS id to build on.
verified_against_classical should be read in that light: the quantum circuit
is verified against a genuine classical computation, but that computation is
not sourced from problem 908 itself.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_qubits: int):
    """Compute the primes in [0, 2**n_qubits) by trial division, from scratch."""
    N = 2 ** n_qubits
    return sorted(x for x in range(N) if is_prime(x))


def build_oracle(n_qubits: int, marked_states):
    """Phase-flip oracle marking each state in marked_states."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")
        # flip qubits that are 0 in this state so the marked pattern becomes all-1
        zero_positions = [i for i, b in enumerate(reversed(bits)) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, marked_states, shots: int = 4096):
    N = 2 ** n_qubits
    M = len(marked_states)
    if M == 0 or M == N:
        raise ValueError("Grover requires 0 < M < N marked states")

    # optimal number of Grover iterations
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4  # search space {0, ..., 15}
    N = 2 ** n_qubits

    classical_answer = classical_primes(n_qubits)
    print(f"Classical primes in [0, {N}): {classical_answer}")

    counts, iterations = run_grover(n_qubits, classical_answer)
    total_shots = sum(counts.values())

    # decode measured bitstrings (Qiskit orders bits with qubit 0 as the
    # rightmost character) back to integers
    measured_hits = {}
    for bitstring, cnt in counts.items():
        value = int(bitstring, 2)
        measured_hits[value] = measured_hits.get(value, 0) + cnt

    hits_on_marked = sum(cnt for v, cnt in measured_hits.items() if v in classical_answer)
    fraction_on_marked = hits_on_marked / total_shots

    # the measured distribution should be strongly concentrated on the
    # classically-computed prime states after amplitude amplification
    most_likely = sorted(measured_hits.items(), key=lambda kv: -kv[1])
    top_values = sorted(v for v, _ in most_likely[: len(classical_answer)])

    print(f"Grover iterations used: {iterations}")
    print(f"Fraction of shots landing on a classical prime: {fraction_on_marked:.3f}")
    print(f"Top {len(classical_answer)} most-measured states: {top_values}")

    verified = (fraction_on_marked > 0.75) and (top_values == classical_answer)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
