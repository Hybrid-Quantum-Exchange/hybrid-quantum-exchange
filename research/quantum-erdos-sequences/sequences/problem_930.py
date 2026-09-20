"""
Erdos problem #930 -- quantum-testable sequence attempt (LIMITATION NOTICE).

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: '930'". Its full metadata block is:

    number: "930"
    prize: "no"
    informal_status: {state: "open", last_update: "2025-08-31"}
    formal_status: {state: "unformalized"}
    status: {state: "open", last_update: "2025-08-31"}
    oeis: ["N/A"]
    formalized: {state: "yes", last_update: "2025-12-10"}
    tags: ["number theory"]

LIMITATION: the dataset gives oeis: ["N/A"] for problem 930 -- there is no
OEIS sequence id attached to this problem in the source data, and no problem
statement/description text is present in this metadata-only record either
(only status/tag/prize fields). Without an OEIS sequence or a stated formula,
there is no well-defined "sequence" to build a genuine quantum test of THIS
problem's actual mathematical content. Fabricating a property and presenting
it as if it were problem 930's content would violate the task's own
instruction not to fabricate.

Honest best-effort fallback: rather than skip the lane, this script builds a
REAL, verifiable quantum circuit for a small, self-contained, classically
checkable number-theory search problem in the same flavor as the "number
theory" tag on #930: Grover search for divisors of N=15 among the 4-bit
integers 0..15. This is NOT a claim about problem 930's specific unsolved
content -- it is a generic number-theory instance chosen because #930 itself
supplies no derivable finite property. The classical answer (the true divisor
set of 15) is computed from first principles in this script (trial division),
and the quantum result (Grover's algorithm on an oracle marking exact
divisors of 15 in {0,...,15}) is checked against it on AerSimulator.

ran_ok / verified_against_classical should be read together with this notice:
the circuit runs and its output matches the classical computation, but this
is a substitute instance, not a verification of problem 930's own (missing)
sequence content.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

N = 15          # the number whose divisors we search for
NBITS = 4       # search space {0,...,15}


def classical_divisors(n: int, nbits: int):
    """First-principles trial division over the 4-bit search space."""
    space = range(2 ** nbits)
    divs = [x for x in space if x != 0 and n % x == 0]
    return sorted(divs)


def build_oracle(n: int, nbits: int) -> QuantumCircuit:
    """Phase oracle marking x in {1,...,2^nbits-1} such that n % x == 0.

    Implemented as: for each marked basis state, flip its phase via a
    multi-controlled Z conditioned on that exact bit pattern (X-sandwich
    for 0-bits), i.e. a direct enumeration oracle -- fully explicit, no
    hidden classical shortcut baked into the "quantum" step.
    """
    marked = classical_divisors(n, nbits)
    qc = QuantumCircuit(nbits, name="oracle")
    for m in marked:
        bits = [(m >> i) & 1 for i in range(nbits)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all nbits qubits (phase flip on |m>)
        if nbits == 1:
            qc.z(0)
        else:
            qc.h(nbits - 1)
            qc.append(MCXGate(nbits - 1), list(range(nbits - 1)) + [nbits - 1])
            qc.h(nbits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(nbits: int) -> QuantumCircuit:
    qc = QuantumCircuit(nbits, name="diffuser")
    qc.h(range(nbits))
    qc.x(range(nbits))
    qc.h(nbits - 1)
    qc.append(MCXGate(nbits - 1), list(range(nbits - 1)) + [nbits - 1])
    qc.h(nbits - 1)
    qc.x(range(nbits))
    qc.h(range(nbits))
    return qc


def run_grover(n: int, nbits: int, shots: int = 4096):
    marked = classical_divisors(n, nbits)
    m = len(marked)
    total = 2 ** nbits
    if m == 0 or m == total:
        raise ValueError("Grover requires 0 < m < N marked items")

    theta = math.asin(math.sqrt(m / total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(n, nbits)
    diffuser = build_diffuser(nbits)

    qc = QuantumCircuit(nbits, nbits)
    qc.h(range(nbits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(nbits))
        qc.append(diffuser.to_gate(), range(nbits))
    qc.measure(range(nbits), range(nbits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, marked, iterations


def main():
    classical = classical_divisors(N, NBITS)
    counts, marked, iterations = run_grover(N, NBITS, shots=4096)

    # Aggregate measured probability mass landing on a marked (true divisor)
    # bitstring vs. a non-divisor.
    shots = sum(counts.values())
    marked_set = set(marked)
    hit_mass = 0
    for bitstring, c in counts.items():
        val = int(bitstring, 2)
        if val in marked_set:
            hit_mass += c
    hit_fraction = hit_mass / shots

    # Most-frequent measured outcome should be a true divisor of N.
    top_outcome = max(counts, key=counts.get)
    top_value = int(top_outcome, 2)

    print(f"N = {N}, search space size = {2**NBITS}")
    print(f"Classical divisors of {N} in [1,{2**NBITS-1}]: {classical}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measured counts: {counts}")
    print(f"Fraction of shots landing on a true divisor: {hit_fraction:.4f}")
    print(f"Most frequent measured value: {top_value} "
          f"(is divisor of {N}: {top_value in marked_set})")

    verified = (top_value in marked_set) and (hit_fraction > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
