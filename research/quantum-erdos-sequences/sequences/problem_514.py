"""
Erdos problem #514 -- quantum-testable-sequence lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: '514'"):
    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["analysis"]

LIMITATION (read before trusting the PASS below):
Problem #514 carries no OEIS sequence id at all (oeis: ["N/A"]) and its tag is
"analysis" rather than a combinatorial/number-theoretic counting problem, so
there is no finite integer sequence attached to it that a small quantum
circuit could search or verify membership in. Fabricating an OEIS id or a
"defining property" for a sequence that does not exist here would violate the
task's own instructions, so this script does NOT test problem #514's actual
mathematical content -- there is none available in finite/computable form.

Honest fallback actually implemented below:
A self-contained, genuinely quantum computation with a classically-checkable
answer, unrelated to any fabricated "problem 514 sequence": Grover's algorithm
searching a 4-qubit space (N = 16) for the marked property "x is prime",
i.e. x in {2, 3, 5, 7, 11, 13}. This is a small, finite, computable property
(primality of a 4-bit integer) with a classical answer computed from first
principles in this script (trial division), and Grover's algorithm is run on
the ideal AerSimulator to recover exactly that same marked set with high
probability. (A 3-qubit version was tried first; primality over {0,...,7}
happens to mark exactly half the space, for which Grover's diffuser is
provably the identity operation and no amplification is possible -- this is
a real, verified property of the algorithm, not a bug, and is the reason for
using 4 qubits instead.) This demonstrates the quantum-testable-sequence
machinery requested by the lane, but the reader should not interpret it as a
statement about Erdos problem #514, which remains open and has no attached
OEIS sequence to test.

Reported accurately: ran_ok reflects whether this script executes and prints
PASS; verified_against_classical reflects whether the quantum measurement
distribution matches the classically-computed marked set -- for the *fallback*
computation described above, not for problem #514 itself, since no such
computation for #514 exists.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_marked_set(n_bits: int):
    N = 2**n_bits
    return sorted(x for x in range(N) if is_prime(x))


def build_oracle(n_bits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle: multi-controlled Z on each marked basis state."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_bits}b")[::-1]  # qubit0 = LSB
        flip_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in flip_qubits:
            qc.x(q)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        for q in flip_qubits:
            qc.x(q)
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


def run_grover(n_bits: int, marked: list[int], shots: int = 4096):
    N = 2**n_bits
    M = len(marked)
    if M == 0 or M == N:
        raise ValueError("Grover requires 0 < M < N marked items")

    # optimal iteration count
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_bits), range(n_bits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_bits = 4  # N = 16, small finite search space (n_bits=3 gives an
    # exact 4/8 marked split, for which Grover's diffuser is provably a
    # no-op -- the marked fraction must differ from 1/2 for amplification)
    classical = classical_marked_set(n_bits)
    print(f"Classical marked set (primes in 0..{2**n_bits - 1}): {classical}")

    counts, iterations = run_grover(n_bits, classical)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    shots = sum(counts.values())
    marked_hits = sum(
        c for bitstring, c in counts.items()
        if int(bitstring, 2) in classical
    )
    fraction_on_marked = marked_hits / shots
    print(f"Fraction of shots landing on classically-marked states: {fraction_on_marked:.3f}")

    # A working Grover search on this instance should concentrate the vast
    # majority of shots on the marked set.
    verified = fraction_on_marked > 0.75

    print()
    print("NOTE: this verifies a fallback Grover-search instance (primality "
          "over 4 bits), not Erdos problem #514, which has no attached OEIS "
          "sequence (oeis: ['N/A']) and hence no finite computable property "
          "to test. See module docstring.")

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    import sys
    sys.exit(0 if ok else 1)
