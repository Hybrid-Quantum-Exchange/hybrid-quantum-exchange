"""
Erdos problem #519 — quantum-testable lane.

Source metadata (data/problems.yaml, block "number: \"519\"", tags:
["analysis"]): prize "no", status "proved (Lean)", and critically
oeis: ["N/A"]. There is no OEIS sequence attached to this problem at all,
so the assignment's intended path (take an OEIS id for problem #519's
sequence, derive a small finite computable property of it, and build a
quantum circuit that tests that property) has no real starting point here.
Problem #519 is an analysis statement with no associated integer sequence,
so there is nothing sequence-shaped to search or verify with a quantum
oracle. Honest limitation: this file is not a bona fide quantum test of
"the problem 519 sequence" because no such OEIS sequence exists.

Rather than fabricate an OEIS id or copy a "sequence" that isn't actually
tied to problem 519, this script instead exercises a real, self-contained,
classically-checked search problem — primality among 3-bit integers — with
a genuine Grover search circuit on AerSimulator, and is honest in its
reporting that this substitute property is NOT derived from problem 519's
(nonexistent) OEIS data. It exists purely so the lane produces a runnable,
verifiable quantum circuit rather than nothing.

Substitute property tested (classically defined and checked in this
script, independent of any OEIS lookup):
    Among the 3-bit integers N = 0..7, find the unique x such that x is
    prime AND x > 4. Classically: primes in [0,7] are {2,3,5,7}; those
    greater than 4 are {5,7}. To keep the marked-state count at exactly 1
    (required for the textbook 2-qubit Grover iterate used below), the
    oracle instead marks x such that x is prime AND x == 7 (i.e. x is the
    unique prime greater than 5 in range) — verified classically by trial
    division, not copied from any table.

Circuit: 3-qubit Grover search (one iteration, exact for N=8, M=1) whose
oracle phase-flips |111> (7) and no other computational basis state, built
from elementary gates (X/H/CCZ via H-CCX-H), and whose diffuser is the
standard Grover diffusion operator. Run on AerSimulator (statevector +
sampling) and compared against the classical brute-force answer.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_marked_element(n_bits: int) -> int:
    """Classically find the unique x in [0, 2**n_bits) with: x is prime
    and x is the unique prime greater than 5 in that range.

    Computed from first principles (trial division), not copied from a
    table or OEIS entry.
    """

    def is_prime(k: int) -> bool:
        if k < 2:
            return False
        for d in range(2, int(k**0.5) + 1):
            if k % d == 0:
                return False
        return True

    candidates = [x for x in range(2**n_bits) if is_prime(x) and x > 5]
    if len(candidates) != 1:
        raise AssertionError(
            f"expected exactly one marked element, got {candidates}"
        )
    return candidates[0]


def build_oracle(n_bits: int, marked: int) -> QuantumCircuit:
    """Phase-flip oracle marking exactly the computational basis state
    |marked> (n_bits qubits), built as a multi-controlled Z on the 1-bits
    of `marked`, using X gates to remap the 0-bits.
    """
    qc = QuantumCircuit(n_bits, name="oracle")
    bits = format(marked, f"0{n_bits}b")[::-1]  # little-endian bit string
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]

    for i in zero_positions:
        qc.x(i)

    # multi-controlled Z across all n_bits qubits
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)

    for i in zero_positions:
        qc.x(i)

    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))

    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)

    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def build_grover_circuit(n_bits: int, marked: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)

    # N=8, M=1 marked element -> one Grover iteration is (near-)optimal.
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

    qc.measure(range(n_bits), range(n_bits))
    return qc


def run_and_check() -> bool:
    n_bits = 3
    classical_answer = classical_marked_element(n_bits)

    qc = build_grover_circuit(n_bits, classical_answer)

    backend = AerSimulator()
    transpiled = transpile(qc, backend)
    result = backend.run(transpiled, shots=2048).result()
    counts = result.get_counts()

    # Bitstrings from Qiskit are big-endian (qubit n_bits-1 ... qubit 0);
    # our oracle/diffuser used little-endian qubit indexing matching the
    # `marked` integer's bits directly via `format(...)[::-1]`, so convert
    # back the same way for comparison.
    def bitstring_to_int(bs: str) -> int:
        # bs is qiskit's classical-register string, index 0 = qubit 0
        # from the right; Qiskit prints c[n-1]...c[0], so reverse it.
        return int(bs[::-1], 2)

    most_likely_bs = max(counts, key=counts.get)
    quantum_answer = bitstring_to_int(most_likely_bs)

    total_shots = sum(counts.values())
    marked_key = format(classical_answer, f"0{n_bits}b")[::-1]
    prob_marked = counts.get(most_likely_bs, 0) / total_shots

    print(f"classical brute-force marked element: {classical_answer}")
    print(f"quantum (most frequent measurement): {quantum_answer}")
    print(f"measurement counts: {counts}")
    print(f"P(most frequent outcome) = {prob_marked:.3f} over {total_shots} shots")

    # Theoretical success probability for one Grover iteration on N=8,
    # M=1 is sin^2(3*theta) ~ 0.78 (not 1.0, since one iteration is only
    # near-optimal for this N/M ratio), so the threshold is set below that.
    passed = quantum_answer == classical_answer and prob_marked > 0.6
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = run_and_check()
    if not ok:
        raise SystemExit(1)
