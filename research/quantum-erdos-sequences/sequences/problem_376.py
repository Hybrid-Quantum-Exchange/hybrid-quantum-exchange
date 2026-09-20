"""
Erdos problem #376 (https://www.erdosproblems.com/376), OEIS A030979.

A030979 is: numbers n such that binomial(2n, n) is NOT divisible by 3.
By Kummer's theorem, the exponent of the prime p in binomial(2n, n) equals
the number of carries when n is added to itself in base p. So binomial(2n,n)
is not divisible by 3 exactly when adding n + n in base 3 produces NO carry,
which happens exactly when every base-3 digit of n is 0 or 1 (a digit of 2
would double to 4 = "1" with carry).

Classical property tested here (computed from first principles, no OEIS
lookup of values): for n in 0..8 (2-digit base-3 numbers, digits d1,d0 with
n = 3*d1 + d0), mark n as "in A030979" iff C(2n,n) mod 3 != 0. We compute
that directly with Python's math.comb, independently of the base-3 digit
argument, and separately confirm the two characterizations agree. Then we
encode n on 4 qubits as two base-3 digits (b3 b2 = d1, b1 b0 = d0, each a
2-bit field, valid values 0..2) and run Grover's algorithm to search for the
"no carry when doubled" states, i.e. members of A030979 restricted to n in
0..8. Because the marked set has 4 out of 16 basis states in this 4-qubit
encoding, the correct number of Grover iterations is 1 (optimal for
M=4, N=16: floor(pi/4 * sqrt(N/M)) = floor(pi/4 * 2) = 1).

The circuit is a real Grover search (Hadamards, phase oracle built from the
classically-derived marked set, diffusion operator) run on Qiskit's ideal
AerSimulator (qasm-style sampling, no noise model). PASS/FAIL compares the
set of basis states with the highest measured probability against the
classical marked set {0, 1, 3, 4}.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


def base3_digits(n: int) -> tuple[int, int]:
    """Return (d1, d0) with n = 3*d1 + d0, 0 <= d0 < 3, for n in 0..8."""
    d0 = n % 3
    d1 = n // 3
    return d1, d0


def classical_membership_via_binomial(n: int) -> bool:
    """True iff C(2n, n) is NOT divisible by 3 (the actual A030979 test)."""
    c = math.comb(2 * n, n)
    return c % 3 != 0


def classical_membership_via_digits(n: int) -> bool:
    """True iff every base-3 digit of n (n in 0..8, two digits) is 0 or 1."""
    d1, d0 = base3_digits(n)
    return d1 in (0, 1) and d0 in (0, 1)


def bits_for_n(n: int) -> tuple[int, int, int, int]:
    """Encode n in 0..8 as 4 qubits: (b0, b1, b2, b3) with
    d0 = 2*b1 + b0, d1 = 2*b3 + b2, n = 3*d1 + d0.
    Only valid for n whose digits are < 3 (always true for n in 0..8)."""
    d1, d0 = base3_digits(n)
    b1, b0 = divmod(d0, 2)
    b3, b2 = divmod(d1, 2)
    return b0, b1, b2, b3


def main() -> None:
    N_RANGE = range(9)  # n = 0..8, all representable with 2 base-3 digits

    # --- Classical ground truth, derived two independent ways ---
    marked_by_binomial = {n for n in N_RANGE if classical_membership_via_binomial(n)}
    marked_by_digits = {n for n in N_RANGE if classical_membership_via_digits(n)}
    assert marked_by_binomial == marked_by_digits, (
        f"digit characterization disagrees with binomial test: "
        f"{marked_by_binomial} vs {marked_by_digits}"
    )
    marked_n = sorted(marked_by_binomial)
    print(f"Classical A030979 membership for n in 0..8 (C(2n,n) mod 3 != 0): {marked_n}")

    # Encode each marked n as a 4-bit basis state string (qubit order b3 b2 b1 b0,
    # Qiskit prints bitstrings with qubit 0 as the rightmost character).
    marked_bitstrings = set()
    for n in marked_n:
        b0, b1, b2, b3 = bits_for_n(n)
        bitstring = f"{b3}{b2}{b1}{b0}"
        marked_bitstrings.add(bitstring)
    print(f"Marked 4-qubit basis states: {sorted(marked_bitstrings)}")

    num_qubits = 4
    num_states = 2**num_qubits
    M = len(marked_bitstrings)

    # --- Build the Grover oracle from the classically-derived marked set ---
    def build_oracle() -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="oracle")
        for bitstring in marked_bitstrings:
            # bitstring[i] is qubit (num_qubits-1-i); flip 0-bits to 1 so an
            # all-ones control pattern corresponds to this basis state.
            zero_qubits = [
                num_qubits - 1 - i for i, ch in enumerate(bitstring) if ch == "0"
            ]
            for q in zero_qubits:
                qc.x(q)
            # Multi-controlled Z on all qubits: phase-flip only this state.
            qc.h(num_qubits - 1)
            qc.append(MCXGate(num_qubits - 1), list(range(num_qubits)))
            qc.h(num_qubits - 1)
            for q in zero_qubits:
                qc.x(q)
        return qc

    def build_diffuser() -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="diffuser")
        qc.h(range(num_qubits))
        qc.x(range(num_qubits))
        qc.h(num_qubits - 1)
        qc.append(MCXGate(num_qubits - 1), list(range(num_qubits)))
        qc.h(num_qubits - 1)
        qc.x(range(num_qubits))
        qc.h(range(num_qubits))
        return qc

    # Optimal number of Grover iterations for M marked out of N states,
    # using the exact formula r = floor(pi / (4*theta)), theta = asin(sqrt(M/N)).
    theta = math.asin(math.sqrt(M / num_states))
    iterations = max(1, math.floor(math.pi / (4 * theta)))
    print(f"N={num_states} states, M={M} marked, Grover iterations={iterations}")

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    oracle = build_oracle()
    diffuser = build_diffuser()
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Sum measured probability mass landing on the marked basis states.
    marked_shots = sum(counts.get(bs, 0) for bs in marked_bitstrings)
    marked_prob = marked_shots / shots
    print(f"Measured probability mass on marked states: {marked_prob:.4f}")

    # Top-M measured outcomes should exactly be the marked set.
    top_states = {bs for bs, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:M]}
    print(f"Top-{M} measured basis states: {sorted(top_states)}")

    ok = (top_states == marked_bitstrings) and (marked_prob > 0.7)

    if ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
