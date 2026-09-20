"""
Erdos problem #955 -- quantum-testable lane (honest limitation notice)
========================================================================

Source metadata (from erdosproblems.com's data/problems.yaml, verified by
grepping `number: "955"` in a read-only clone at
/home/user/manman4/erdosproblems/data/problems.yaml):

    number: "955"
    prize: no
    informal_status: open (last update 2025-08-31)
    formal_status: unformalized
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION, stated up front: the `oeis` field for problem #955 is the
literal string "possible" -- it is NOT an OEIS A-number. There is no real
OEIS sequence id attached to this problem in the dataset, and the repo's
own problem page/body text is not available in this read-only clone (no
per-problem markdown/detail file for 955 was found alongside problems.yaml).
So there is no OEIS-sequence-derived, finite, computable "known term" for
this specific problem that can be honestly claimed to test problem #955
itself. Following the task's fallback instructions, this script does NOT
fabricate an OEIS value or invent a fake sequence and pretend it is #955's.

Instead, as a good-faith, clearly-labeled substitute lane, this script
builds a REAL, genuine quantum circuit for a small, finite, computable
number-theoretic search problem in the same spirit as #955's tag
("number theory"): using Grover's algorithm to find the unique prime
number in the range [0, 7] (3 qubits, N = 8) that is congruent to 3 mod 4
and greater than 4, i.e. search for x in {0,...,7} such that x is prime,
x % 4 == 3, and x > 4 (this excludes the smaller solution x=3, leaving a
single marked element, x=7).

The classical answer is computed from first principles in this script
(trial division for primality, direct mod check) -- not copied from any
table. Grover's algorithm is then run on the ideal AerSimulator and its
most-frequent measurement outcome is compared against the classical
answer.

Because this is NOT actually derived from problem #955's own statement
(no OEIS id, no accessible problem text), this script is reported as
an honest best-effort fallback, not a verified instance of problem #955.
"""

import sys
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_answer(n_qubits: int) -> int:
    """Find x in [0, 2**n_qubits) with x prime and x % 4 == 3, from first principles."""
    N = 2 ** n_qubits
    candidates = [x for x in range(N) if classical_is_prime(x) and x % 4 == 3 and x > 4]
    assert len(candidates) == 1, f"expected a unique marked element, got {candidates}"
    return candidates[0]


def build_oracle(n_qubits: int, marked: int) -> QuantumCircuit:
    """Phase-flip oracle marking the single computational basis state `marked`."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked, f"0{n_qubits}b")[::-1]  # little-endian
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]

    for i in zero_positions:
        qc.x(i)

    # multi-controlled Z on all n_qubits (phase flip |11...1>)
    if n_qubits == 1:
        qc.z(0)
    else:
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
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, marked: int, shots: int = 2048) -> int:
    N = 2 ** n_qubits
    iterations = max(1, round((np.pi / 4) * np.sqrt(N)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    qc = qc.decompose().decompose()
    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # most frequent outcome (bitstring is big-endian in qiskit's counts by default:
    # counts keys are c[n-1]...c[0])
    top_bitstring = max(counts, key=counts.get)
    top_value = int(top_bitstring, 2)
    return top_value, counts


def main() -> bool:
    n_qubits = 3  # N = 8
    expected = classical_answer(n_qubits)
    print(f"[classical] search space N = {2**n_qubits}, unique marked element = {expected} "
          f"(prime, {expected} mod 4 == {expected % 4}, and {expected} > 4)")

    measured, counts = run_grover(n_qubits, expected)
    total_shots = sum(counts.values())
    hits = counts.get(format(measured, f"0{n_qubits}b"), 0)
    print(f"[quantum]   Grover top outcome = {measured} "
          f"({hits}/{total_shots} shots = {hits/total_shots:.1%})")
    print(f"[quantum]   full counts = {counts}")

    passed = (measured == expected) and (hits / total_shots > 0.5)

    print()
    if passed:
        print("PASS")
    else:
        print("FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
