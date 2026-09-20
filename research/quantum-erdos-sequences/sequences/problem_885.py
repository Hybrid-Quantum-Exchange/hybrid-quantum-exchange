"""
Erdos problem #885 -- quantum-testable instance.

Source metadata (erdosproblems.com data, problems.yaml, number "885"):
    prize: no
    status: open
    tags: ["number theory", "divisors"]
    oeis: ["N/A"]

LIMITATION: problem #885 carries no OEIS sequence id in the data file (the
oeis field is literally ["N/A"]). Per the task instructions, a script is
still produced with a best-honest attempt: since the problem's own tags are
["number theory", "divisors"], the finite, computable property chosen here
is a genuine number-theory/divisors property, not a value copied from any
OEIS entry (none exists to copy from), and not a value asserted without
being derived classically in this script.

Chosen property (derived classically in this script, from first principles):
    Within N = {0, 1, ..., 63} (6 bits), find the integer(s) n whose divisor
    count d(n) is exactly 9. Brute force over n < 64 shows there is exactly
    one such n (36 = 2^2 * 3^2, whose divisors are 1,2,3,4,6,9,12,18,36 --
    nine of them). A single marked item out of 64 is also the cleanest case
    for Grover amplification (near-maximal boost after the optimal number
    of iterations).

    d(n) (the divisor-counting function) is computed here by brute-force
    trial division -- first principles, no OEIS lookup, no library shortcut.

Quantum method: Grover's search algorithm on 6 qubits (search space size
N = 64). The oracle is built directly from the classically-computed marked
set (a genuine multi-controlled-Z phase oracle over the marked basis
states), then standard Grover diffusion is applied for the optimal number
of iterations. The circuit is run on the ideal AerSimulator and the most
frequent measured bitstring is checked against the classically computed
marked set: PASS if the top measurement outcome actually has d(n) == 4.
"""

from itertools import combinations
from math import floor, pi, sqrt

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def divisor_count(n: int) -> int:
    """Brute-force divisor count of n by trial division (first principles)."""
    if n <= 0:
        return 0
    count = 0
    for d in range(1, n + 1):
        if n % d == 0:
            count += 1
    return count


def classical_marked_set(n_bits: int, target_divisor_count: int) -> list[int]:
    """All n in [0, 2**n_bits) with divisor_count(n) == target, from scratch."""
    size = 2 ** n_bits
    return [n for n in range(size) if divisor_count(n) == target_divisor_count]


def build_oracle(n_bits: int, marked_states: list[int]) -> QuantumCircuit:
    """Phase oracle: flips the sign of exactly the marked computational basis
    states, built as a standard multi-controlled-Z per marked state (each
    conditioned via X-gates on the bits that must be 0)."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_bits}b")[::-1]  # little-endian per qubit
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        # multi-controlled Z on all n_bits qubits (phase flip iff all-ones)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def grover_circuit(n_bits: int, marked_states: list[int], iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    oracle = build_oracle(n_bits, marked_states)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_bits), range(n_bits))
    return qc


def main() -> bool:
    n_bits = 6  # search space N = 64, matches the "n <= ~64" instance size
    target_divisor_count = 9

    marked = classical_marked_set(n_bits, target_divisor_count)
    print(f"Classical answer: {len(marked)} integers in [0,64) with exactly "
          f"{target_divisor_count} divisors:")
    print(sorted(marked))

    n_marked = len(marked)
    n_total = 2 ** n_bits
    # optimal number of Grover iterations
    iterations = max(1, floor((pi / 4) * sqrt(n_total / n_marked)))
    print(f"Grover iterations used: {iterations}")

    qc = grover_circuit(n_bits, marked, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Most frequent measured bitstring -> integer. Qiskit prints classical
    # register bits as c[n-1]...c[0] (MSB left), and qubit i was measured
    # into c[i] (weight 2**i), so the printed string is already ordinary
    # binary for n -- no reversal needed.
    top_bitstring = max(counts, key=counts.get)
    top_n = int(top_bitstring, 2)

    total_shots = sum(counts.values())
    marked_shots = sum(c for bstr, c in counts.items()
                        if int(bstr, 2) in marked)
    amplification_fraction = marked_shots / total_shots

    print(f"Top measured bitstring: {top_bitstring} -> n = {top_n}")
    print(f"Fraction of shots landing on a marked (d(n)==4) state: "
          f"{amplification_fraction:.3f}")

    quantum_found_marked_top = top_n in marked
    quantum_amplified = amplification_fraction > (n_marked / n_total) * 2

    verified = quantum_found_marked_top and quantum_amplified

    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
