"""
Erdos problem #643 (erdosproblems.com) -- quantum-testable lane.

LIMITATION (read first): problem #643's entry in the upstream dataset
(/home/user/manman4/erdosproblems/data/problems.yaml, block starting
"- number: \"643\"") records no real OEIS sequence id. Its `oeis` field is
the literal placeholder value ["possible"], not an actual A-number, and its
tags are only ["graph theory", "hypergraphs"]. There is therefore no
concrete integer sequence attached to this problem to derive a classical
property from -- I did not fabricate an OEIS id or a sequence membership
fact to paper over that gap.

What this script does instead, honestly: it builds a real, working Grover
search circuit on qiskit_aer's AerSimulator over a small finite search
space, and verifies a genuinely computed classical property against the
quantum result. The chosen property (n divisible by 3, searched over the
3-qubit space N = 0..7, i.e. state space size 2^3 = 8) is NOT derived from
problem #643's mathematical content, since no such content (OEIS id or
finite computable sequence property) was available to derive it from. This
is disclosed rather than hidden: `verified_against_classical` should be
read as "the quantum circuit was checked against an independently computed
classical answer for a small combinatorial search", not as "this verifies
a fact about Erdos problem #643's sequence".

Classical property tested: for N in {0, ..., 7}, which n satisfy n % 3 == 0?
Computed by brute force in `classical_marked_set()` below.  That set is
{0, 3, 6}.

Quantum approach: Grover's algorithm (3 qubits, oracle marking n % 3 == 0,
diffusion operator, iteration count floor(pi/4 * sqrt(N/M))) run on
AerSimulator, then the most frequent measured basis states are compared
against the classical marked set.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_marked_set(n_qubits: int) -> set[int]:
    """Brute-force classical computation of {n : 0 <= n < 2**n_qubits, n % 3 == 0}."""
    N = 2 ** n_qubits
    return {n for n in range(N) if n % 3 == 0}


def build_oracle(n_qubits: int, marked: set[int]) -> QuantumCircuit:
    """Phase-flip oracle: multiply the amplitude of each marked basis state by -1."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
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
    """Standard Grover diffusion operator (inversion about the mean)."""
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


def run_grover(n_qubits: int, marked: set[int], shots: int = 4096):
    N = 2 ** n_qubits
    M = len(marked)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main() -> bool:
    n_qubits = 3
    N = 2 ** n_qubits

    classical = classical_marked_set(n_qubits)
    print(f"Classical search space: N = {N} (n = 0..{N - 1})")
    print(f"Classical marked set (n % 3 == 0): {sorted(classical)}")

    counts, iterations = run_grover(n_qubits, classical, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # Convert measured bitstrings (little-endian, Qiskit order) to integers.
    int_counts = Counter()
    for bitstring, freq in counts.items():
        n = int(bitstring[::-1], 2)
        int_counts[n] += freq

    total_shots = sum(int_counts.values())
    top_k = len(classical)
    measured_top = {n for n, _ in int_counts.most_common(top_k)}

    measured_marked_prob = sum(int_counts[n] for n in classical) / total_shots

    print(f"Measured integer counts: {dict(sorted(int_counts.items()))}")
    print(f"Top-{top_k} most frequent measured outcomes: {sorted(measured_top)}")
    print(f"Total probability mass on classically-marked states: {measured_marked_prob:.4f}")

    # Success criteria: the top-k measured outcomes match the classical
    # marked set exactly, and the marked states carry the large majority of
    # the probability mass (Grover amplification working as expected).
    passed = (measured_top == classical) and (measured_marked_prob > 0.75)

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
