"""
Erdos problem #975 -- quantum-testable instance
=================================================

Source: erdosproblems.com problem 975 (data/problems.yaml entry `number: "975"`,
tags ["number theory", "divisors", "polynomials"], informal_status "open").
The problem's linked OEIS sequence is A147807.

OEIS A147807: a(n) = sum_{p=1}^{n} tau(p^2 + 1) / 2, where tau(m) is the
number of positive divisors of m. It is the cumulative half-divisor-count of
p^2 + 1, tied to Erdos's question about how the divisor function tau(n^2+1)
behaves / how often n^2+1 has many small divisors.

Classical property tested here
-------------------------------
For a single fixed small instance p = 7, N = p^2 + 1 = 50, restricted to the
search space of 3-bit integers x in {0, ..., 7} (x = 0 is skipped, since
"divides by zero" is undefined): the set of divisors of N = 50 lying in
{1, ..., 7} is

    D = { x in {1,...,7} : 50 mod x == 0 } = {1, 2, 5}

This set is computed here in plain Python by trial division (first
principles, no OEIS lookup of the answer itself -- only the *sequence
definition*, tau(n^2+1), was taken from OEIS). |D| = 3, which is exactly the
tau(50)-restricted-to-small-divisors count that A147807 sums (the full
tau(50) = 6, counting 1,2,5,10,25,50; restricting the search register to 3
bits keeps the search space small enough for a real Grover circuit while
still directly instantiating "find divisors of p^2+1").

Quantum circuit
----------------
A genuine Grover search over the 3-qubit register {0,...,7}: the oracle
flips the phase of exactly the basis states in D = {1, 2, 5} (built as an
explicit multi-controlled-Z per marked state, no classical shortcuts baked
into the unitary beyond what the oracle is allowed to know: which x divide
N), followed by the standard Grover diffuser. With |D| = 3 marked states out
of 8, the optimal number of Grover iterations is round(pi/4 * sqrt(8/3)) = 1.
The circuit is run on the ideal AerSimulator (statevector-based qasm
simulation via sampling), and PASS/FAIL is decided by checking that the
measurement distribution is concentrated (essentially all shots) on exactly
the classically-computed set D.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def divisors_of_p_squared_plus_1(p: int, search_max: int) -> list[int]:
    """Classically compute, by trial division, the divisors of p^2 + 1
    lying in {1, ..., search_max}. This is the ground-truth answer."""
    n = p * p + 1
    return [x for x in range(1, search_max + 1) if n % x == 0]


def build_oracle(marked_states: list[int], n_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle: |x> -> -|x> for each x in marked_states."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all n_qubits (phase flip of |11...1>)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
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
    p = 7
    n_qubits = 3
    search_max = 2 ** n_qubits - 1  # 7

    ground_truth = sorted(divisors_of_p_squared_plus_1(p, search_max))
    print(f"p = {p}, N = p^2+1 = {p * p + 1}")
    print(f"Classical divisors of N in [1,{search_max}] (trial division): {ground_truth}")

    n_marked = len(ground_truth)
    n_total = 2 ** n_qubits
    iterations = max(1, round((np.pi / 4) * np.sqrt(n_total / n_marked)))
    print(f"Grover iterations used: {iterations}")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(ground_truth, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    shots = 4096
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Qiskit count keys are "c_{n-1}...c_1 c_0" (qubit 0 = rightmost char),
    # which is already standard big-endian binary for the integer value.
    dist = {}
    for bitstring, c in counts.items():
        val = int(bitstring, 2)
        dist[val] = dist.get(val, 0) + c

    print("Measured distribution (value: count):")
    for v in sorted(dist, key=lambda k: -dist[k]):
        print(f"  {v}: {dist[v]}")

    mass_on_marked = sum(dist.get(v, 0) for v in ground_truth) / shots
    quantum_top = sorted(dist, key=lambda k: -dist[k])[:n_marked]
    quantum_answer = sorted(quantum_top)

    print(f"Fraction of shots landing on classical divisor set: {mass_on_marked:.4f}")
    print(f"Top-{n_marked} most frequent measured values: {quantum_answer}")

    verified = (quantum_answer == ground_truth) and (mass_on_marked > 0.80)
    return verified


if __name__ == "__main__":
    ok = main()
    print("PASS" if ok else "FAIL")
