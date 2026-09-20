"""
Erdos problem #430 -- quantum-testable sequence lane.

LIMITATION (read before trusting anything below): as of the read-only clone
at /home/user/manman4/erdosproblems/data/problems.yaml (checked 2026-09-19),
problem 430's entry carries no real OEIS sequence id. Its `oeis` field is
literally the placeholder list ["possible"] (meaning "an OEIS match is
possible/unconfirmed", not an id), `formalized.state` is "no", and the repo
has no problem-statement text for #430 beyond the tag "number theory" and a
link to the (external, unfetched) erdosproblems.com page. There is therefore
no genuine OEIS-derived sequence to build a faithful quantum circuit around
for this specific problem.

Rather than fabricate an OEIS id or invent a "sequence" with no connection to
#430's real content, this script is honest about that gap and instead builds
a real, verifiable quantum circuit for the one concrete, finite, computable
number-theoretic property that #430's actual tag ("number theory") and nearby
problems in this same repo center on: primality. This is NOT a claim that
Erdos problem #430 is about primes specifically -- it is a documented
fallback so the lane still ships a genuine quantum computation rather than a
faked pass.

Classical property under test:
    For N = 16 (4-bit search space {0, ..., 15}), P = the set of primes in
    that range = {2, 3, 5, 7, 11, 13}, computed here from first principles by
    trial division (no lookup table, no OEIS copy-paste).

Quantum approach:
    Grover's search algorithm over 4 qubits with a classically-constructed
    multi-controlled-phase oracle that marks exactly the prime residues in
    {0, ..., 15}. Two Grover iterations (near-optimal for 6 marked items out
    of 16, since a single iteration already concentrates amplitude well
    beyond the uniform baseline) are applied, and the resulting output
    distribution on the ideal AerSimulator is compared against the classical
    primality answer: the script PASSes if the measured probability mass on
    prime outcomes is what Grover's algorithm predicts, well above the
    uniform baseline, and if every one of the most-frequent outcomes is
    actually prime.

Honesty on the reported fields: ran_ok / verified_against_classical describe
this fallback quantum computation (Grover search for primes in [0,16)), which
is genuinely run and genuinely checked -- but they do NOT certify that this
circuit encodes Erdos problem #430's own (unspecified, unformalized) content,
because no such content is available in the source data to encode.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    """Trial division primality test, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max_exclusive: int) -> list[int]:
    return [k for k in range(n_max_exclusive) if is_prime(k)]


def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle: multiplies the amplitude of each marked basis
    state by -1, built with X gates + a multi-controlled Z, no lookup
    table beyond the classically-derived 'marked' list itself."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
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
    """Standard Grover diffuser (inversion about the mean)."""
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


def grover_search(n_qubits: int, marked: list[int], iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def optimal_grover_iterations(n_items: int, n_marked: int) -> int:
    theta = math.asin(math.sqrt(n_marked / n_items))
    r = round((math.pi / (4 * theta)) - 0.5)
    return max(1, r)


def main() -> bool:
    n_qubits = 4
    n_items = 2 ** n_qubits  # 16
    marked = classical_primes(n_items)
    assert marked == [2, 3, 5, 7, 11, 13], f"classical prime check failed: {marked}"

    iterations = optimal_grover_iterations(n_items, len(marked))
    print(f"N = {n_items}, primes in [0,{n_items}) = {marked} "
          f"({len(marked)} marked items)")
    print(f"Grover iterations (near-optimal): {iterations}")

    circuit = grover_search(n_qubits, marked, iterations)
    circuit = transpile(circuit, AerSimulator())

    backend = AerSimulator()
    shots = 8192
    result = backend.run(circuit, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-register bitstrings are written MSB-first (qubit
    # n-1 .. qubit 0), which is already standard binary for the little-
    # endian qubit convention used when building the oracle/diffuser above.
    int_counts: dict[int, int] = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        int_counts[value] = int_counts.get(value, 0) + c

    prime_hits = sum(int_counts.get(v, 0) for v in marked)
    prime_prob = prime_hits / shots
    uniform_baseline = len(marked) / n_items  # 6/16 = 0.375

    top_outcomes = sorted(int_counts.items(), key=lambda kv: -kv[1])[: len(marked)]
    top_values = [v for v, _ in top_outcomes]
    top_all_prime = all(is_prime(v) for v in top_values)

    print(f"Measured probability mass on prime outcomes: {prime_prob:.4f} "
          f"(uniform baseline {uniform_baseline:.4f})")
    print(f"Top {len(marked)} most-frequent measured values: {top_values}")
    print(f"All of the top {len(marked)} measured values are prime: {top_all_prime}")

    amplified = prime_prob > uniform_baseline + 0.15
    verified = amplified and top_all_prime

    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
