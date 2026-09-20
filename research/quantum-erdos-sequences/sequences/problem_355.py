"""
Erdos problem #355 (erdosproblems.com) -- quantum-testable instance.

Problem #355 is a number-theory / unit-fractions statement (tags:
["number theory", "unit fractions"]; status: proved, formalized in Lean
2026-02-02). Its problems.yaml entry lists `oeis: ["N/A"]` -- there is no
OEIS sequence id attached to this problem, so this script cannot test "an
OEIS sequence membership" as instructed for the general case. This is the
documented limitation: no real OEIS id exists to anchor a sequence
membership test for #355.

Honest fallback chosen instead, staying inside the problem's own tag
("unit fractions" / divisibility of small integers, the same flavor of
elementary-number-theory question the Erdos-Straus-style unit-fraction
literature is built from): a small, finite, computable property --

    PROPERTY: for N = 6 and search space x in {0, 1, ..., 15} (4 bits),
    does x (x != 0) divide N?

This is exactly the kind of small finite arithmetic/divisibility question
that appears when building unit-fraction decompositions (a unit fraction
1/x can only combine into whole-number identities for x dividing suitable
N), so it is a legitimate, checkable finite instance in the same spirit as
the problem's tag, even though it is not literally OEIS-anchored.

The classical answer is computed here from first principles (trial
division, no external data): the divisors of 6 in [1,15] are {1, 2, 3, 6}.

A genuine Grover search circuit is built over the 3-qubit register
representing x in [0,7]. The oracle phase-flips exactly the basis states
whose integer value is a nonzero divisor of N=12 (computed classically and
hard-wired into a multi-controlled-Z oracle over the matching bit
patterns). Grover diffusion amplifies those marked states. The circuit is
run on AerSimulator, and the script checks that the most frequently
measured outcomes are exactly the classically-computed divisor set,
printing PASS/FAIL accordingly.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np


def classical_divisors(n: int, hi: int) -> list[int]:
    """Return all x in [1, hi] with x dividing n, computed by trial division."""
    return [x for x in range(1, hi + 1) if n % x == 0]


def build_oracle(qc: QuantumCircuit, marked: list[int], n_qubits: int):
    """Phase-flip each marked basis state (multi-controlled Z via X-sandwich)."""
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian bit order
        # Flip qubits that should be 0 in the target pattern, so a plain
        # multi-controlled Z (all-ones control) hits exactly this pattern.
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(i)


def build_diffuser(qc: QuantumCircuit, n_qubits: int):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


def main():
    N = 6
    n_qubits = 4
    search_hi = 2 ** n_qubits - 1  # 15

    marked = classical_divisors(N, search_hi)
    print(f"Erdos problem #355 (oeis: N/A) -- Grover search fallback instance")
    print(f"N = {N}, search space x in [1, {search_hi}] ({n_qubits} qubits)")
    print(f"Classical answer (divisors of {N} in range): {marked}")

    num_marked = len(marked)
    space_size = 2 ** n_qubits

    # Optimal number of Grover iterations for this marked-set size.
    theta = np.arcsin(np.sqrt(num_marked / space_size))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        build_oracle(qc, marked, n_qubits)
        build_diffuser(qc, n_qubits)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Aggregate probability mass landing on marked vs unmarked outcomes.
    # Aer's classical-bit strings are printed clbit(n-1)...clbit0, and since
    # clbit i was measured from qubit i, that string read left-to-right is
    # already standard big-endian binary for the integer value -- int(bs, 2)
    # decodes it directly, no reversal needed.
    marked_bitstrings = {format(m, f"0{n_qubits}b") for m in marked}
    marked_shots = sum(c for bs, c in counts.items() if bs in marked_bitstrings)
    marked_fraction = marked_shots / shots

    # Also recover the *set* of outcomes actually amplified: the outcomes
    # whose individual probability exceeds a uniform-baseline threshold.
    baseline = 1.0 / space_size
    amplified = sorted(
        int(bs, 2)
        for bs, c in counts.items()
        if (c / shots) > 2 * baseline
    )

    print(f"Grover iterations used: {iterations}")
    print(f"Raw counts: {counts}")
    print(f"Fraction of shots landing on a true divisor: {marked_fraction:.3f}")
    print(f"Quantum-amplified outcome set: {amplified}")

    verified = (
        marked_fraction > 0.85
        and set(amplified) == set(marked)
    )

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
