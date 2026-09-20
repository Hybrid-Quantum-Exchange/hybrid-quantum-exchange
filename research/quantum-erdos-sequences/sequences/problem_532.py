"""
Erdos problem #532 -- quantum-testable companion script.

Erdos problem #532 (per data/problems.yaml in the erdosproblems repository,
https://github.com/manman4/erdosproblems) is a number-theory / Ramsey-theory
statement recorded as proved (Lean-formalized). Its metadata entry carries
`oeis: ["N/A"]` -- there is NO OEIS sequence id attached to this problem, and
no finite, computable "is this integer in the sequence" question is defined
by the problem record itself. That means the requested workflow (derive a
small property of *the* OEIS sequence for #532, then build a quantum circuit
that computes/verifies it) cannot be done faithfully: there is no sequence to
target.

LIMITATION, stated plainly: this script does not test any property that is
actually tied to Erdos problem #532's mathematical content beyond its
declared tags (["number theory", "ramsey theory"]). Fabricating a specific
numeric claim and presenting it as "the #532 sequence" would misrepresent
the source data. Instead, as the best honest fallback, this script builds a
REAL, correct quantum circuit for a small, finite, computable number-theory
search problem in the same tag family as #532 -- Grover's algorithm search
for prime numbers in the 4-bit range 0..15 -- and verifies the quantum
result against a classical primality check computed from first principles
(trial division) in this same script. This is a genuine quantum computation
with correct classical verification; it is NOT a verification of Erdos
problem #532 itself, and should not be read as one.

Property tested: for N = 16 (4 qubits, integers 0..15), which integers are
prime? Grover's algorithm is used to amplify the amplitude of the prime
marked states {2, 3, 5, 7, 11, 13} out of the 16 basis states (6/16 marked,
which avoids the degenerate 1/2-marked case where a single Grover iteration
leaves the uniform distribution unchanged), and the post-amplification
measurement distribution is checked against the classically-computed prime
set.

Classical answer (computed below by trial division, not copied from OEIS):
primes in [0, 15] = {2, 3, 5, 7, 11, 13}
"""

import sys
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, XGate


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max_exclusive: int) -> set:
    return {n for n in range(n_max_exclusive) if is_prime(n)}


def build_oracle(n_qubits: int, marked_states: list) -> QuantumCircuit:
    """Phase-flip oracle: multiplies the amplitude of each marked basis
    state (given as little-endian bitstrings) by -1, via a multi-controlled Z
    implemented as X-sandwiched MCZ per marked state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCMTGate(XGate(), n_qubits - 1, 1), list(range(n_qubits)))
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCMTGate(XGate(), n_qubits - 1, 1), list(range(n_qubits)))
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_prime_search(n_qubits: int, marked_states: list, shots: int = 4096):
    n_total = 2 ** n_qubits
    n_marked = len(marked_states)
    # Optimal number of Grover iterations for this marked-count / space-size ratio.
    theta = np.arcsin(np.sqrt(n_marked / n_total))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4
    n_total = 2 ** n_qubits  # search space {0,...,15}

    expected_primes = classical_primes(n_total)
    marked_states = sorted(expected_primes)

    counts, iterations = grover_prime_search(n_qubits, marked_states)

    # Little-endian bitstrings -> integers, aggregate counts per integer.
    shots_total = sum(counts.values())
    per_state_counts = {n: 0 for n in range(n_total)}
    for bitstring, c in counts.items():
        # Qiskit's classical-register bitstrings are written with clbit 0
        # (= qubit 0, measured by qc.measure(range(n), range(n))) as the
        # rightmost character -- i.e. already the standard binary string
        # with qubit 0 as the least-significant bit. int(bitstring, 2) is
        # exactly the integer value that was searched for.
        value = int(bitstring, 2)
        per_state_counts[value] += c

    # The states measured most frequently should be exactly the marked
    # (prime) states, since Grover amplifies their amplitude.
    threshold = shots_total / n_total  # uniform-baseline count per state
    observed_amplified = {
        n for n, c in per_state_counts.items() if c > threshold
    }

    print(f"Search space: integers 0..{n_total - 1} ({n_qubits} qubits)")
    print(f"Classical primes (trial division): {sorted(expected_primes)}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measured counts per integer (shots={shots_total}): {per_state_counts}")
    print(f"States amplified above uniform baseline: {sorted(observed_amplified)}")

    verified = observed_amplified == expected_primes
    print(f"Quantum result matches classical primes: {verified}")

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
