"""
Erdos problem #848 -- quantum-testable companion script.

Source metadata (from erdosproblems.com data, as cloned to
/home/user/manman4/erdosproblems/data/problems.yaml, entry "number: \"848\""):
    prize: no
    status: decidable (as of 2025-10-19)
    oeis: ["N/A"]   -- no OEIS sequence id is attached to this problem
    tags: ["number theory"]

LIMITATION (reported honestly, per instructions): the data file gives no
OEIS id and no problem statement text for #848 (only the tag "number
theory" and a "decidable" status), so there is no specific integer
sequence from #848 to build a faithful quantum test of. This script is my
best honest attempt at a small, real, quantum-testable number-theory
property in that spirit, rather than a fabricated tie-in to a sequence
that doesn't exist in the source data.

Chosen property (finite, classically verifiable, and a legitimate quantum
search target): "n is prime" for n in {0, ..., 15} (a 4-qubit search
space). The classical answer -- the set of primes below 16 -- is computed
here from first principles by trial division, not copied from anywhere.

Quantum approach: Grover's algorithm. A phase oracle is built directly
from the classically-computed prime set (multi-controlled-Z on the basis
states corresponding to primes, i.e. a literal encoding of the classical
predicate "is prime" into circuit gates), followed by the standard
Grover diffuser, run on the ideal AerSimulator. This is a genuine
amplitude-amplification computation: it starts from a uniform
superposition over all 4-bit strings and amplifies exactly the marked
(prime) basis states using the correct number of Grover iterations for
N=16, M=|primes<16| marked items.

PASS criterion: after running the circuit and measuring with many shots,
the set of basis states with non-negligible measured probability must
equal exactly the classically-computed prime set {2,3,5,7,11,13}, and
each individual prime state's measured probability must exceed a
generous threshold showing genuine amplification (uniform baseline is
1/16 = 6.25%; amplified probability should be well above that).

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_primes_below(n: int) -> list[int]:
    """Compute primes in [0, n) by trial division, from first principles."""
    primes = []
    for candidate in range(2, n):
        is_prime = True
        for d in range(2, int(math.isqrt(candidate)) + 1):
            if candidate % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
    return primes


def build_oracle(num_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Phase oracle flipping the sign of each marked computational basis state."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for state in marked_states:
        zero_positions = [i for i in range(num_qubits) if not (state >> i) & 1]
        for i in zero_positions:
            qc.x(i)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def main() -> bool:
    n = 16
    num_qubits = 4
    classical_primes = classical_primes_below(n)
    print(f"Classical primes below {n} (trial division): {classical_primes}")

    m = len(classical_primes)
    # Optimal number of Grover iterations for N items, M marked.
    iterations = max(1, round((math.pi / 4) * math.sqrt(n / m) - 0.5))
    print(f"N={n}, M={m}, Grover iterations={iterations}")

    oracle = build_oracle(num_qubits, classical_primes)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit classical-bit strings are "c_{n-1}...c_1 c_0" (rightmost char is
    # bit 0 = qubit 0's result), so interpreting the string directly as
    # binary already gives the integer basis state with qubit i as bit i.
    probs = {}
    for bitstring, count in counts.items():
        state = int(bitstring, 2)
        probs[state] = probs.get(state, 0.0) + count / shots

    print("Measured probabilities per basis state:")
    for state in sorted(probs):
        print(f"  n={state:2d}: p={probs[state]:.4f}")

    threshold = 2.0 / n  # well above the uniform baseline of 1/16
    measured_marked = sorted(s for s, p in probs.items() if p >= threshold)

    verified = measured_marked == sorted(classical_primes)
    print(f"Quantum-identified 'prime' states (p >= {threshold:.3f}): {measured_marked}")
    print(f"Classical prime states: {sorted(classical_primes)}")

    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
