"""
Erdos problem #264 -- quantum-testable lane.

Source metadata (from erdosproblems.com data, manman4/erdosproblems
data/problems.yaml, entry "number: \"264\"", tags: ["irrationality"]):
    oeis: ["N/A"]

LIMITATION (reported honestly, per task instructions): problem #264 has no
OEIS sequence attached (oeis == "N/A"). It concerns irrationality of a
real-valued constant/series, which is not a finite, computable membership
or search property that a small quantum circuit can meaningfully test --
there is no finite instance whose "correct answer" is the actual open
problem. Rather than fabricate a fake tie-in to problem #264, this script
falls back to a genuine, self-contained, classically-verified finite
search problem in the same spirit as problems in the "irrationality" /
number-theory family that Erdos-problem sequences typically reduce to:

    Property tested: "which integers n in {0, 1, ..., 7} are perfect
    squares?" (i.e. n = k^2 for some non-negative integer k). This is
    exactly the kind of small, finite, decidable set-membership property
    that Erdos-adjacent OEIS entries (e.g. A000290, squares) are built
    from, and it is computed classically from first principles below
    (by brute-force squaring), then verified with a real Grover search
    circuit on the ideal AerSimulator.

No OEIS id was used, because none exists for problem #264. This script is
therefore this lane's "best honest attempt": ran_ok reflects whether the
circuit executes, and verified_against_classical reflects whether the
quantum result matches the independently computed classical answer for
the fallback finite search problem -- NOT a verification of anything
about Erdos problem #264 itself.
"""

import itertools
import sys

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate

N_QUBITS = 3  # search space {0,...,7}


def classical_perfect_squares(n_qubits: int):
    """Brute-force, from first principles: which n in [0, 2^n_qubits) are
    perfect squares (n == k*k for some integer k >= 0)."""
    size = 2 ** n_qubits
    marked = []
    for n in range(size):
        k = int(round(n ** 0.5))
        if k * k == n:
            marked.append(n)
    return sorted(marked)


def build_oracle(n_qubits: int, marked_values):
    """Phase-flip oracle: negates the amplitude of each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, marked_values, shots: int = 2048):
    import math

    size = 2 ** n_qubits
    m = len(marked_values)
    if m == 0:
        raise ValueError("no marked values -- Grover needs at least one solution")

    # Optimal number of Grover iterations for m marked items out of size.
    iterations = max(1, round((math.pi / 4) * math.sqrt(size / m)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    marked = classical_perfect_squares(N_QUBITS)
    print(f"Classical answer (perfect squares in [0, {2**N_QUBITS})): {marked}")

    counts, iterations = run_grover(N_QUBITS, marked, shots=2048)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    total_shots = sum(counts.values())
    marked_bitstrings = {format(v, f"0{N_QUBITS}b")[::-1] for v in marked}
    hits = sum(c for bs, c in counts.items() if bs in marked_bitstrings)
    hit_fraction = hits / total_shots

    # Quantum-measured "answer": the set of outcomes that individually
    # exceed a naive-uniform-noise threshold (i.e. the amplified states).
    threshold = (1.0 / (2 ** N_QUBITS)) * total_shots * 2  # 2x uniform baseline
    quantum_marked = sorted(
        int(bs[::-1], 2) for bs, c in counts.items() if c > threshold
    )

    print(f"Fraction of shots landing on a true perfect square: {hit_fraction:.3f}")
    print(f"Quantum-inferred marked set: {quantum_marked}")

    verified = (quantum_marked == marked) and (hit_fraction > 0.8)

    if verified:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
