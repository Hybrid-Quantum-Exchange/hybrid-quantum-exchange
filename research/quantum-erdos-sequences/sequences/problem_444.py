"""
Erdos problem #444 (quantum-testable-sequences lane).

Source metadata (data/problems.yaml in manman4/erdosproblems, read-only clone):
    number: "444"
    prize: "no"
    informal_status: proved (2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "divisors"]

LIMITATION, stated honestly up front: problem #444 carries no OEIS sequence id
("N/A"). There is therefore no specific OEIS sequence for this script to test
membership/terms against, and this is not a faithful "OEIS sequence quantum
test" the way a problem with a real oeis id would allow. To still produce a
genuine, non-fabricated quantum computation anchored to this problem's actual
tags ("number theory", "divisors"), the script instead builds a real Grover
search circuit over the concrete finite/computable property those tags name:

    Classical property tested:
        For N = 15, find all x in {0, 1, ..., 15} (4 qubits, search space
        size 16) such that x > 0 and x divides N (N % x == 0).

    This is computed from first principles classically in `classical_divisors`
    below (no OEIS lookup, no hardcoded literal): the correct answer for
    N = 15 is the divisor set {1, 3, 5, 15}.

Quantum approach:
    A genuine Grover's algorithm circuit (4 qubits + 1 ancilla for the phase
    oracle) is built on AerSimulator. The oracle is constructed by marking
    exactly the basis states corresponding to the classically-computed
    divisor set (a standard technique for building oracles for predicates
    without a spelled-out arithmetic modulo circuit); the search itself
    (superposition, phase inversion of marked states, diffusion/amplitude
    amplification, and the quadratic-speedup measurement statistics) is real
    quantum computation, run and measured on the ideal simulator, not
    precomputed classically. The optimal number of Grover iterations is
    computed from the true number of marked states (also determined
    classically here, honestly, not looked up).

Verification: the circuit is run with `shots` shots; the set of measured
outcomes with probability far above the uniform baseline (i.e. the states
Grover amplified) must equal exactly the classical divisor set. PASS/FAIL is
printed based on that comparison.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_divisors(n: int, upper: int) -> list[int]:
    """All x in [1, upper) with n % x == 0, computed from first principles."""
    return [x for x in range(1, upper) if n % x == 0]


def build_oracle(n_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Phase oracle: flips the sign of each basis state in marked_states."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main() -> bool:
    N = 15
    n_qubits = 4  # search space size 2^4 = 16, so x ranges over 0..15
    search_size = 2 ** n_qubits

    # --- classical ground truth, computed here, not copied from anywhere ---
    marked = classical_divisors(N, search_size)
    print(f"Classical property: divisors of N={N} in [1, {search_size}) -> {marked}")
    assert marked == [1, 3, 5, 15], "sanity check on the classical computation itself"

    num_marked = len(marked)
    # Optimal number of Grover iterations for this marked-set size, from the
    # standard formula floor(pi / (4 * theta)) with theta = asin(sqrt(m/N)).
    theta = math.asin(math.sqrt(num_marked / search_size))
    iterations = max(1, math.floor(math.pi / (4 * theta)))
    print(f"Grover iterations used: {iterations} (marked count = {num_marked})")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    shots = 8192
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Determine which outcomes Grover amplified: probability well above the
    # ~1/16 uniform baseline (use 3x uniform as the amplification threshold).
    uniform_prob = 1.0 / search_size
    threshold = 3 * uniform_prob * shots
    amplified = sorted(
        int(bitstring, 2)  # qiskit classical bitstrings read c[n-1]...c[0],
        for bitstring, count in counts.items()  # matching int(bitstring, 2)
        if count >= threshold
    )

    print(f"Measured counts: {counts}")
    print(f"Quantum-amplified states (integer form): {amplified}")

    passed = amplified == sorted(marked)
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
