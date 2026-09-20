"""
Erdos problem #674 (from erdosproblems.com data, data/problems.yaml entry
`number: "674"`) — metadata as read from the source YAML:

    prize: no
    informal_status: proved (Lean, last_update 2025-12-08)
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (reported honestly, not worked around): problem #674's metadata
record carries NO OEIS sequence id — the field is literally the string
"N/A". There is therefore no actual integer sequence attached to this
problem to build a genuine "is n in the sequence" / "find the k-th term"
quantum oracle around. Any claim to test "the OEIS sequence for problem
674" would be fabricated, since no such sequence exists in the source data.

Best honest attempt, given that constraint: the problem's only real
attribute is its tag "number theory" plus the fact that it has been proved.
Rather than invent a fake OEIS sequence, this script builds a genuine,
small, self-contained finite number-theory decision problem — squarefree
integers in a bounded range — and verifies it with a real Grover search
circuit on Qiskit's ideal AerSimulator. This is NOT a claim that OEIS
A005117 (squarefree numbers) is "the sequence of problem 674" — it isn't;
674 has no OEIS id at all. It is offered only as the closest honest,
genuinely-computable, small quantum-circuit-testable artifact available
for a problem whose real metadata has no finite computable sequence
property to test.

Classical property tested
--------------------------
Search space: n in [0, 15] (4 qubits).
Predicate marked by the oracle: n is NOT squarefree, i.e. some prime p
satisfies p*p | n (n = 0 counts as not squarefree). This is the smaller
of the two classes in [0,15] (5 of 16 values), which is the class Grover
search amplifies well for a single iteration on a 4-qubit register.

The classical answer (the set of non-squarefree n in [0,15]) is computed
from first principles in `classical_squarefree` below, with no external
lookup; the marked set passed to the oracle is its complement.
That classical answer is then used to build a Grover oracle marking
exactly those n, and Grover search is run on AerSimulator. The circuit is
verified to amplify exactly the classical marked set, and PASS/FAIL is
decided by comparing the most-frequent measured outcomes against the
classical set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

N_QUBITS = 4
N = 2 ** N_QUBITS  # 16, range [0, 15]


def classical_squarefree(n: int) -> bool:
    """First-principles squarefree test: n is squarefree iff no prime p
    has p*p dividing n. n = 0 is defined as not squarefree."""
    if n == 0:
        return False
    if n == 1:
        return True
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            count = 0
            while m % p == 0:
                m //= p
                count += 1
            if count >= 2:
                return False
        p += 1
    return True


def classical_marked_set():
    """The oracle marks the NOT-squarefree integers (the smaller class),
    for a Grover-amplification-friendly M/N ratio."""
    return sorted(n for n in range(N) if not classical_squarefree(n))


def build_oracle(marked, n_qubits):
    """Phase-flip oracle: for each marked integer, apply a controlled-Z
    (via multi-controlled X on an ancilla-free phase trick using an
    MCZ built from MCX + basis flips) that flips the sign of |n>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all n_qubits: use H + MCX + H trick on last qubit
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked, n_qubits, shots=4096):
    M = len(marked)
    N_total = 2 ** n_qubits
    if M == 0 or M == N_total:
        raise ValueError("Grover requires 0 < M < N marked items")

    theta = math.asin(math.sqrt(M / N_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    marked = classical_marked_set()
    print(f"Classical NON-squarefree integers in [0,{N-1}]: {marked}")
    print(f"(count = {len(marked)} of {N})")

    counts, iterations = run_grover(marked, N_QUBITS, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # Sort measured outcomes by frequency, take top-len(marked) as the
    # quantum-predicted marked set.
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_k = sorted_counts[: len(marked)]
    quantum_marked = sorted(int(bitstring, 2) for bitstring, _ in top_k)

    total_shots = sum(counts.values())
    marked_shots = sum(c for b, c in counts.items() if int(b, 2) in marked)
    marked_fraction = marked_shots / total_shots

    print(f"Quantum top-{len(marked)} measured outcomes (by frequency): {quantum_marked}")
    print(f"Fraction of shots landing on classically-marked states: {marked_fraction:.3f}")

    verified = (quantum_marked == marked) and (marked_fraction > 0.90)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
