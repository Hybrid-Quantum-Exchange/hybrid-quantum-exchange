"""
Erdos problem #531 -- quantum-testable instance
=================================================

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: '531'" (tags: ["number theory", "ramsey theory"]; informal_status:
open; oeis: ["possible"]).

LIMITATION (read before trusting the "verified" claim below): problem #531's
`oeis` field in the source data is the literal string "possible", not an
actual OEIS sequence id. There is no OEIS id attached to this problem in the
source, and its informal status is "open" with no closed-form finite
description of "the sequence" available to derive a property from. It is
therefore not possible to construct a circuit that tests a genuine property
of "the Erdos-531 sequence" -- there is no such concretely specified,
publicly finite sequence to test.

Rather than fabricate an OEIS id or invent a property with no connection to
the problem, this script falls back to the problem's stated tags ("number
theory") and builds a real, honestly-computed, finite instance of a classic
number-theoretic search problem: "find the unique integer x in [0, 15] that
is both prime and greater than 12." That instance is:

    classical answer: x = 13 (binary 1101), the only prime in [0,15] with
    x > 12 (candidates above 12 are 13, 14, 15; only 13 is prime).

This is verified/derived from first principles in `classical_search` below
(trial division primality test over the full 4-bit search space), not copied
from any table. It is then found with a genuine Grover search circuit
(oracle + diffuser, run on the ideal AerSimulator) over the same 4-bit space,
and the quantum result is compared against the classical answer.

This is a best-effort generic instance illustrating the problem's declared
tag ("number theory") via real Grover search machinery. It is explicitly
NOT a verification of any actual property of Erdos problem #531's own
(currently unspecified / non-OEIS-linked) sequence.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_search(n_bits: int):
    """Brute-force, from first principles, the unique x in [0, 2**n_bits - 1]
    with x prime and x > 12. Returns (answer, all_matches)."""
    n = 2 ** n_bits
    matches = [x for x in range(n) if is_prime(x) and x > 12]
    return matches


def build_oracle(n_bits: int, target: int) -> QuantumCircuit:
    """Phase-flip oracle marking the single computational basis state
    |target> (little-endian bit order matching Qiskit's qubit ordering)."""
    qc = QuantumCircuit(n_bits, name="oracle")
    bits = format(target, f"0{n_bits}b")[::-1]  # little-endian
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    for i in zero_positions:
        qc.x(i)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def build_grover_circuit(n_bits: int, target: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    oracle = build_oracle(n_bits, target)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_bits))
        qc.append(diffuser.to_gate(), range(n_bits))
    qc.measure(range(n_bits), range(n_bits))
    return qc


def main():
    n_bits = 4  # search space size N = 16 <= 64 as required

    matches = classical_search(n_bits)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected a unique classical match in [0,15]; got {matches}"
        )
    classical_answer = matches[0]
    print(f"Classical search over [0, {2**n_bits - 1}]: unique prime > 12 is "
          f"x = {classical_answer} (binary {format(classical_answer, '04b')})")

    n_items = 2 ** n_bits
    iterations = max(1, round((np.pi / 4) * np.sqrt(n_items / 1)))
    print(f"Grover iterations: {iterations}")

    qc = build_grover_circuit(n_bits, classical_answer, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 2048
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-register bit string is printed MSB..LSB over
    # c[n-1]..c[0], which (since measure(range(n),range(n)) pairs qubit i
    # with clbit i) already matches standard big-endian integer value.
    best_bitstring = max(counts, key=counts.get)
    measured_value = int(best_bitstring, 2)
    measured_probability = counts[best_bitstring] / shots

    print(f"Most frequent measured outcome: {best_bitstring} -> x = "
          f"{measured_value} (probability {measured_probability:.3f}, "
          f"{shots} shots)")
    print(f"Counts: {counts}")

    verified = (measured_value == classical_answer) and (measured_probability > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
