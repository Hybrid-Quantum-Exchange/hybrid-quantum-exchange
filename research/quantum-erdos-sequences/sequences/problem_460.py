"""
Erdos problem #460 (erdosproblems.com/460) -- quantum-testable-sequence lane.

LIMITATION (read this first): problem #460's entry in erdosproblems/data/
problems.yaml carries oeis: ["N/A"] and comments: "ambiguous statement".
There is no OEIS sequence attached to this problem, so no property of "the
sequence for problem 460" can honestly be derived or tested -- there is no
such sequence in the data. This script is therefore NOT a genuine quantum
test of anything belonging to problem 460. Per the task's fallback
instructions ("write the script anyway with your best honest attempt, note
the limitation clearly"), what follows is a small, self-contained, genuinely
quantum computation of an unrelated but well-defined and verifiable
classical number-theoretic property, built with a real Grover search on
AerSimulator, so the file still demonstrates the intended mechanics
(classical ground truth computed from first principles, oracle circuit,
comparison, PASS/FAIL). It should NOT be read as evidence about problem 460
or about any of its (nonexistent) OEIS ids.

Chosen substitute property (search space N = 16, i.e. 4 qubits):
    Find the unique x in {0, ..., 15} such that x is divisible by 3 AND
    x is divisible by 5 (i.e. x is a multiple of 15, restricted to the
    4-bit range). Classically this is x = 0 and x = 15 -- two solutions --
    so to keep the search space small and the marked-state count exactly 1
    (needed for a clean, high-probability 2-qubit-oracle Grover instance)
    we search over {1, ..., 15} by simply excluding 0 from consideration
    at the classical-answer stage and building the oracle to mark only
    x = 15 (the unique nonzero multiple of 15 below 16).

The classical answer is computed here in the script, from first
principles (trial division), not copied from any table.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_answer(n_bits: int) -> int:
    """Return the unique x in [1, 2**n_bits - 1] divisible by both 3 and 5,
    computed by direct trial division (first principles)."""
    n = 1 << n_bits
    candidates = [x for x in range(1, n) if x % 3 == 0 and x % 5 == 0]
    assert len(candidates) == 1, f"expected exactly one marked state, got {candidates}"
    return candidates[0]


def build_oracle(n_bits: int, marked: int) -> QuantumCircuit:
    """Phase-flip oracle marking the single computational basis state
    |marked> among n_bits qubits (multi-controlled Z with 0-controls
    flipped via X gates around unset bits)."""
    qc = QuantumCircuit(n_bits, name="oracle")
    bits = [(marked >> i) & 1 for i in range(n_bits)]
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
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


def grover_find(n_bits: int, marked: int, shots: int = 2048) -> int:
    n = 1 << n_bits
    # Optimal number of Grover iterations for a single marked item.
    iterations = max(1, round((np.pi / 4) * np.sqrt(n)))

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_bits))
        qc.append(diffuser.to_instruction(), range(n_bits))
    qc.measure(range(n_bits), range(n_bits))
    qc = qc.decompose()

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit ordering: classical bit c[0] is the rightmost character.
    best_bitstring = max(counts, key=counts.get)
    measured = int(best_bitstring[::-1], 2)
    return measured, counts


def main() -> None:
    n_bits = 4
    expected = classical_answer(n_bits)
    print(f"Classical answer (first-principles trial division): x = {expected}")

    measured, counts = grover_find(n_bits, expected)
    total_shots = sum(counts.values())
    top_prob = counts[max(counts, key=counts.get)] / total_shots
    print(f"Grover measured most-frequent result: x = {measured} "
          f"(probability {top_prob:.3f} over {total_shots} shots)")
    print(f"Counts histogram: {counts}")

    passed = (measured == expected) and (top_prob > 0.5)
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
