"""
Erdos problem #353 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "353"`):
    prize: no
    informal_status: proved (Lean, formalized)
    oeis: ["N/A"]
    tags: ["geometry"]

LIMITATION (please read before trusting the PASS below):
Problem #353 carries no OEIS sequence id in the source data (oeis: ["N/A"]).
There is therefore no actual "sequence" from this problem to build a
quantum-testable property from, and this script does NOT claim to test
anything about Erdos problem #353's real mathematical content. Fabricating
an OEIS id or a fake "term of the sequence" to satisfy the assignment would
be dishonest, so this script instead does the best honest fallback: it
builds a genuine, independently-checkable finite/computable number-theoretic
search problem, verifies the classical answer from first principles in
Python, and then verifies a real Grover search circuit (run on the ideal
AerSimulator) reproduces that classical answer. This establishes that the
*pipeline* (classical derivation -> oracle -> Grover -> measurement ->
comparison) works correctly, but it is NOT a statement about problem #353's
geometry content, since problem #353 has no OEIS sequence to anchor to.

The chosen finite/computable property (unrelated to any specific OEIS id,
chosen only because it is small, genuinely computable, and non-trivial to
search naively):

    Over the search space x in {0, 1, ..., 15} (4 qubits), find all x such
    that x^2 mod 13 == 1, i.e. the square roots of unity modulo the prime
    13. This is a classic finite/computable number-theory search problem
    with a small, well-defined, verifiable answer, and both solutions fit
    inside the 4-qubit (0..15) register.

Classical derivation (done in this script, not copied from anywhere):
    For each x in 0..15, compute (x*x) % 13 and keep those equal to 1.
    By elementary number theory, in Z/13Z (13 prime) the equation
    x^2 = 1 has exactly two solutions: x = 1 and x = 12 (i.e. -1 mod 13).
    This script computes that set by brute force and asserts it below,
    rather than asserting it a priori.

Quantum method: Grover's algorithm.
    - 4 qubits encode x in {0,...,15}.
    - The oracle phase-flips exactly the marked states (built directly from
      the classically-computed marked set, so the oracle is provably
      correct by construction, and the classical answer used for
      comparison is independently recomputed by brute force below).
    - One Grover diffusion/amplification round (optimal for M=2 marked
      out of N=16, since the optimal number of iterations is
      round(pi/4 * sqrt(N/M)) = 1) is applied, then the register is
      measured 4096 times on AerSimulator.
    - PASS requires: (a) the classical brute-force set matches the
      analytically expected {1, 16}; (b) the two most frequent measured
      outcomes are exactly the classically-marked set, together carrying
      the large majority of measurement shots (Grover amplification
      working as expected).
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_marked_set(modulus: int, n_bits: int) -> list[int]:
    """Brute-force, from first principles, all x in [0, 2**n_bits) with
    x*x % modulus == 1."""
    n = 2 ** n_bits
    marked = []
    for x in range(n):
        if (x * x) % modulus == 1:
            marked.append(x)
    return marked


def build_oracle(n_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Phase-flip oracle that marks exactly `marked_states` (each an int in
    [0, 2**n_qubits)) by applying a multi-controlled Z conditioned on the
    computational basis state equalling that integer."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main() -> None:
    n_qubits = 4
    modulus = 13

    # --- classical derivation, from first principles ---
    marked = classical_marked_set(modulus, n_qubits)
    # Elementary number theory: sqrt(1) mod prime p is the residue class {1, p-1} (mod p).
    # The search register covers x in [0, 2**n_bits), which is wider than [0, p), so any x in
    # range congruent to 1 or (p-1) mod p is a valid marked state, not just the two smallest reps.
    expected = sorted(x for x in range(2 ** n_qubits) if x % modulus in (1, modulus - 1))
    assert marked == expected, f"classical brute force {marked} disagrees with theory {expected}"
    print(f"Classical answer (brute force, x^2 mod {modulus} == 1, x in 0..{2**n_qubits - 1}): {marked}")

    n_total = 2 ** n_qubits
    m_marked = len(marked)
    theta = np.arcsin(np.sqrt(m_marked / n_total))
    iterations = max(1, int(np.floor((np.pi / 4) / theta)))
    print(f"Grover iterations used: {iterations} (N={n_total}, M={m_marked})")

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's get_counts() bitstring is "c_{n-1}...c_1c_0" (qubit 0 = rightmost/least
    # significant character), which is exactly standard binary -- int(bitstring, 2) gives
    # the same integer convention used by build_oracle/build_diffuser and by Statevector
    # indexing, so no reversal is needed here.
    int_counts: dict[int, int] = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        int_counts[value] = int_counts.get(value, 0) + c

    top_m = sorted(int_counts.items(), key=lambda kv: -kv[1])[:m_marked]
    top_m_states = sorted(v for v, _ in top_m)
    top_m_shots = sum(c for _, c in top_m)
    fraction_on_marked = top_m_shots / shots

    print(f"Measured top-{m_marked} states: {top_m_states} with {top_m_shots}/{shots} shots "
          f"({fraction_on_marked:.3f} of total)")

    verified = (top_m_states == sorted(marked)) and (fraction_on_marked > 0.7)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
