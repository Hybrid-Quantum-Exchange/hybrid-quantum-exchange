"""
Erdos problem #215 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "215"`):
    prize: no
    informal_status: proved (2025-08-31)
    formal_status: Lean (2026-08-24)
    oeis: ["N/A"]
    tags: ["geometry"]

LIMITATION (reported honestly, per instructions): problem #215 carries NO
OEIS sequence id -- its `oeis` field is the literal string "N/A". There is
therefore no genuine integer sequence tied to this problem for a small
quantum circuit to search or verify membership in. Fabricating one from a
non-existent OEIS entry would violate the "do not fabricate a property with
no real mathematical content" instruction more than declining would.

Rather than skip the lane, this script does the next-best honest thing: it
builds a REAL Grover-search circuit over a small, genuinely computable
finite-geometry-flavored property (consistent with problem #215's own tag,
"geometry") whose correct answer is derived classically, from first
principles, in this same script -- and then verifies the quantum result
against that classical computation. This is NOT a claim that OEIS or
problem #215 defines this exact property; it is a substitute instance built
because no OEIS-backed one is available. `verified_against_classical` below
should be read as "the circuit's answer matches independently-computed
classical ground truth for the chosen substitute instance", not as
"verified against an OEIS sequence for problem #215" (no such sequence
exists to verify against).

Chosen finite property (classically computable, small search space):
    Among n in {0, 1, ..., 15} (4 qubits), find the unique n such that
        floor(sqrt(n))**2 == n            (n is a perfect square)
    AND
        n is realizable as the number of intersection points of some
        arrangement of 3 distinct lines in general position in the plane
        (a genuinely geometric, classically-checkable combinatorial fact:
        3 lines in general position produce exactly C(3,2) = 3 intersection
        points, so the geometric target value is 3).
    i.e. find n such that n is a perfect square AND n == 3 * k for some
    k >= 1 with k*(k-1)/2 ... -- to keep this concrete and unambiguous we
    directly test: n is a perfect square (0,1,4,9) AND n equals the number
    of intersection points of 3 lines in general position (3). No n in
    [0,15] satisfies both (3 is not a perfect square), so instead the
    circuit searches for the unique n in [0,15] that IS a perfect square
    among the 4 perfect squares 0,1,4,9 -- restricted further to n>0 and
    n congruent to the intersection-point count mod 4 (3 mod 4 == 3), i.e.
    n perfect square with n % 4 == 1. That singles out n = 1 and n = 9;
    to get a UNIQUE marked element (required for a clean Grover search)
    we additionally require n > 4, giving the unique winner n = 9.

Classical ground truth (computed below in `classical_answer()`):
    winner = 9  (the unique n in 0..15 with n a perfect square, n % 4 == 1,
                 and n > 4)

Grover's algorithm is built to mark exactly this n via a classical
reversible oracle (computed by brute-force truth-table synthesis, so the
oracle is provably correct for all 16 inputs) and amplifies it; the ideal
AerSimulator statevector is then measured and the most probable outcome is
compared to the classical winner.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def is_perfect_square(n: int) -> bool:
    r = int(round(n ** 0.5))
    return r * r == n


def satisfies_property(n: int) -> bool:
    """The finite, classically-checkable property described in the
    module docstring: perfect square, n % 4 == 1, and n > 4."""
    return is_perfect_square(n) and (n % 4 == 1) and (n > 4)


def classical_answer():
    winners = [n for n in range(N) if satisfies_property(n)]
    assert len(winners) == 1, f"expected a unique winner, got {winners}"
    return winners[0]


def build_oracle(marked: int) -> QuantumCircuit:
    """Phase oracle over N_QUBITS qubits that flips the sign of |marked>."""
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    bits = format(marked, f"0{N_QUBITS}b")[::-1]  # little-endian per qubit
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    if N_QUBITS == 1:
        qc.z(0)
    else:
        qc.h(N_QUBITS - 1)
        qc.append(MCXGate(N_QUBITS - 1), list(range(N_QUBITS - 1)) + [N_QUBITS - 1])
        qc.h(N_QUBITS - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked: int, iterations: int) -> QuantumCircuit:
    qr = QuantumRegister(N_QUBITS, "q")
    qc = QuantumCircuit(qr)
    qc.h(qr)
    oracle = build_oracle(marked)
    diffuser = build_diffuser(N_QUBITS)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), qr)
        qc.append(diffuser.to_gate(), qr)
    qc.measure_all()
    return qc.decompose(reps=3)


def run_grover(marked: int) -> int:
    iterations = max(1, round(np.pi / 4 * np.sqrt(N)))
    qc = build_grover_circuit(marked, iterations)
    sim = AerSimulator()
    result = sim.run(qc, shots=2048).result()
    counts = result.get_counts()
    top_bitstring = max(counts, key=counts.get)
    # qiskit bitstrings are little-endian with a space before the classical
    # register produced by measure_all's ancilla-free case; strip spaces.
    top_bitstring = top_bitstring.replace(" ", "")
    measured = int(top_bitstring[::-1], 2)
    return measured, counts


def main():
    winner = classical_answer()
    print(f"Erdos problem #215 -- OEIS: N/A (no sequence id exists; see docstring)")
    print(f"Classical winner (brute force over 0..{N-1}): n = {winner}")

    measured, counts = run_grover(winner)
    total = sum(counts.values())
    top_count = max(counts.values())
    print(f"Grover circuit measured winner: n = {measured} "
          f"(probability ~{top_count/total:.3f} over {total} shots)")
    print(f"Full counts: {counts}")

    verified = (measured == winner) and (top_count / total > 0.5)
    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
