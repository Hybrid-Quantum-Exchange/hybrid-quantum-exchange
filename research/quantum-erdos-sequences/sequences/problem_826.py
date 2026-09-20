"""
Erdos problem #826 -- quantum-testable sequence lane.

LIMITATION (read first): Problem #826's entry in erdosproblems/data/problems.yaml
has oeis: ["N/A"] -- there is no OEIS sequence id attached to this problem, and
the dataset carries no informal statement text for it either (only prize/status/
tags: ["number theory"], status "open" as of 2025-08-31). With no sequence and no
problem statement in hand, there is no real property of Erdos problem #826 itself
that this script can test on a quantum circuit -- constructing one would mean
fabricating content the task explicitly forbids.

Rather than fake a pass against a nonexistent sequence, this script honestly
documents that gap and falls back to the best-effort substitute allowed by the
task instructions: a small, finite, genuinely computable number-theoretic
property (squarefreeness), in the spirit of the problem's "number theory" tag,
implemented as a real Grover search circuit and checked against a from-scratch
classical computation. This is NOT a verification of Erdos problem #826 or of
any of its OEIS sequences (none exist in the source data) -- it is a generic,
honestly-labeled quantum number-theory demo standing in for the missing lane
content.

Property actually tested: "N is squarefree" (not divisible by any p^2, p prime)
for N in {0, 1, ..., 7} (3-bit search space).

Classical squarefree set for N in 0..7, computed from first principles below:
    0 -> not squarefree (divisible by every square, by convention excluded)
    1 -> squarefree (empty product)
    2 -> squarefree
    3 -> squarefree
    4 -> not squarefree (4 = 2^2)
    5 -> squarefree
    6 -> squarefree (2*3)
    7 -> squarefree
So the squarefree marked set is {1, 2, 3, 5, 6, 7} (6 of 8 values).

Circuit: 3-qubit Grover search whose oracle marks exactly the squarefree
values in {0,...,7}, computed via a phase oracle built from the classical
squarefree marking (derived in-script, not copied from any table). Because
6 of the 8 basis states are marked, a single Grover iteration would overshoot;
we instead run Grover with 0 iterations' worth of amplification analytically
inapplicable, so instead we directly verify via the *phase oracle plus
diffusion for a deliberately restricted 2-marked subsearch*: search for the
two NON-squarefree values in {0,...,7} (0 and 4), which is the well-posed,
non-degenerate Grover instance the task calls for (few marked items),
computed and marked from first principles in-script.

PASS/FAIL: quantum Grover output (most frequent measured state(s)) compared
against the classically computed set of non-squarefree N in 0..7.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def is_squarefree(n: int) -> bool:
    """Classical, from-scratch squarefreeness check for a small non-negative int."""
    if n <= 0:
        return False
    d = 2
    m = n
    while d * d <= m:
        if m % (d * d) == 0:
            return False
        while m % d == 0:
            m //= d
        d += 1
    return True


def classical_non_squarefree(limit: int):
    return sorted(n for n in range(limit) if not is_squarefree(n))


def build_oracle(marked_states, n_qubits):
    """Phase oracle flipping the sign of exactly the given basis states."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    n_qubits = 3
    limit = 2 ** n_qubits  # search space N in {0,...,7}

    marked = classical_non_squarefree(limit)  # expect [0, 4]
    print(f"Classical non-squarefree values in 0..{limit - 1}: {marked}")

    n_marked = len(marked)
    N = limit
    # Optimal number of Grover iterations for n_marked out of N items.
    theta = np.arcsin(np.sqrt(n_marked / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Interpret measured bitstrings (Qiskit returns MSB..LSB in the string,
    # matching qubit order 0 = rightmost char) back into integers.
    measured_ints = {int(bstr, 2): c for bstr, c in counts.items()}
    sorted_measured = sorted(measured_ints.items(), key=lambda kv: -kv[1])
    top_states = sorted([s for s, _ in sorted_measured[:n_marked]])

    total_shots = sum(counts.values())
    marked_hits = sum(c for s, c in measured_ints.items() if s in marked)
    marked_fraction = marked_hits / total_shots

    print(f"Grover iterations used: {iterations}")
    print(f"Top {n_marked} measured state(s): {top_states}")
    print(f"Fraction of shots landing on a truly non-squarefree state: {marked_fraction:.3f}")

    verified = (top_states == marked) and (marked_fraction > 0.7)

    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    main()
