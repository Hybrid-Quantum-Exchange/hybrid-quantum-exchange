"""
Erdos problem #527 -- quantum-testable-sequence attempt.

LIMITATION (read first): as recorded in erdosproblems.com's own data
(data/problems.yaml, entry `number: "527"`), problem #527 is tagged
["analysis", "probability"] and its `oeis` field is `["N/A"]` -- there is no
OEIS integer sequence attached to this problem at all. The problem itself
(informal_status: proved, 2025-09-08) concerns an analysis/probability
statement, not a combinatorial sequence of integers, so there is no finite,
computable membership/counting/search property of "the sequence for #527"
to build a Grover oracle or phase-estimation circuit around -- because there
is no sequence.

Rather than fabricate an OEIS id or invent a "property" with no connection
to problem 527, this script is a best-honest-effort placeholder: it builds
and runs a REAL, correctly-verified Grover search circuit on a small finite
instance (4 qubits, N=16 candidates, searching for the perfect squares in
[0, 15] = {0, 1, 4, 9}, verified classically by brute force in this script).
This demonstrates genuine quantum search infrastructure that COULD be
retargeted at a real OEIS-backed property, but the property being tested
here (perfect-squareness of a 4-bit integer) is NOT derived from problem
#527's mathematical content, because #527 furnishes no such content to
derive it from.

Reported accurately: ran_ok=True, but verified_against_classical should be
understood as "the placeholder circuit matches its own classical brute
force", not as a verification of any Erdos-527-specific sequence, since no
such sequence exists in the source data.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


def classical_perfect_squares(n_bits: int) -> list[int]:
    """Brute-force, first-principles: which integers in [0, 2**n_bits - 1]
    are perfect squares?"""
    upper = 2 ** n_bits
    result = []
    for x in range(upper):
        r = math.isqrt(x)
        if r * r == x:
            result.append(x)
    return result


def build_oracle(n_bits: int, targets: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each target computational basis state."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for t in targets:
        bits = format(t, f"0{n_bits}b")[::-1]  # little-endian per qubit
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
            qc.h(n_bits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover(n_bits: int, targets: list[int], shots: int = 2048):
    n_states = 2 ** n_bits
    n_marked = len(targets)
    # optimal number of Grover iterations (standard closed form)
    theta = math.asin(math.sqrt(n_marked / n_states))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(n_bits, targets)
    diffuser = build_diffuser(n_bits)

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_bits))
        qc.append(diffuser.to_gate(), range(n_bits))
    qc.measure(range(n_bits), range(n_bits))

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_bits = 4  # search space [0, 15]
    targets = classical_perfect_squares(n_bits)
    print(f"Classical perfect squares in [0, {2**n_bits - 1}]: {targets}")

    counts, iterations = run_grover(n_bits, targets, shots=2048)
    print(f"Grover iterations used: {iterations}")

    # top measured outcomes (bitstrings are big-endian qubit order from Qiskit)
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_n = sorted_counts[: len(targets)]
    print("Top measured outcomes (bitstring: count):")
    for bs, c in sorted_counts[:8]:
        print(f"  {bs}: {c}")

    measured_values = set()
    for bs, _ in top_n:
        # Qiskit's classical-register bitstring already reads as a normal
        # big-endian binary integer (qubit 0 -> rightmost character).
        val = int(bs, 2)
        measured_values.add(val)

    classical_set = set(targets)
    passed = measured_values == classical_set

    print(f"Classical answer: {sorted(classical_set)}")
    print(f"Quantum (Grover) top-{len(targets)} measured values: {sorted(measured_values)}")
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
