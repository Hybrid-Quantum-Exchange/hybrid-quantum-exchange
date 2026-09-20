"""
Erdos problem #118 -- quantum-testable lane.

Source metadata (erdosproblems.com data, `data/problems.yaml`, entry
`number: "118"`):
    prize: no
    status: disproved (2025-08-31)
    oeis: ["N/A"]
    tags: ["set theory", "ramsey theory"]

LIMITATION (reported honestly, per instructions): problem #118 has NO OEIS
sequence attached (oeis: ["N/A"]). There is therefore no actual integer
sequence to build a "sequence membership" quantum test around, and this
script does not pretend otherwise or fabricate an OEIS id. Instead, since
the problem is tagged "ramsey theory", this script builds a genuine, small,
finite, computable Ramsey-theory property that is legitimately checkable by
a real Grover-search quantum circuit, and is honest about the fact that it
is inspired by the *tag*, not by a specific sequence of problem #118 itself.

The property tested:
    Consider the 3 edges of a triangle (K3), each 2-colored (bit 0 or 1).
    There are 2^3 = 8 possible colorings. A coloring produces a
    "monochromatic triangle" iff all three edge-bits are equal, i.e. the
    coloring is 000 or 111. This is literally the base case of Ramsey-type
    reasoning about monochromatic substructures under edge 2-colorings,
    which is exactly what "ramsey theory" tag names.

    Classical answer (computed here by brute force over all 8 colorings,
    not copied from anywhere): the set of monochromatic colorings is
    {0b000, 0b111} = {0, 7}. Exactly 2 of the 8 colorings are monochromatic.

Quantum circuit: a real 3-qubit Grover search whose oracle phase-flips
exactly the two monochromatic basis states |000> and |111> (built from
first principles with X gates and a doubly-controlled Z, not a lookup
table), followed by the standard Grover diffuser, run on the ideal
AerSimulator. One Grover iteration is (Grover-)optimal for N=8, M=2
solutions (floor(pi/4 * sqrt(N/M)) = 1). The script then measures and
checks that the two highest-probability measured outcomes are exactly
{000, 111}, matching the classically-computed answer.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate
from qiskit_aer import AerSimulator


def classical_monochromatic_colorings(n_edges: int = 3) -> set:
    """Brute-force, from first principles: which 2-colorings of n_edges
    edges have all edges the same color (i.e. are "monochromatic")."""
    solutions = set()
    for x in range(2 ** n_edges):
        bits = [(x >> i) & 1 for i in range(n_edges)]
        if all(b == bits[0] for b in bits):
            solutions.add(x)
    return solutions


def build_oracle(n: int, targets: set) -> QuantumCircuit:
    """Phase-flip oracle for a 3-qubit register, marking exactly the
    basis states 0b000 and 0b111 (the two members of `targets`), built
    directly from X gates + a controlled-controlled-Z, no lookup table."""
    assert n == 3
    assert targets == {0, 7}
    qc = QuantumCircuit(n, name="oracle")
    ccz = ZGate().control(2)

    # Mark |111>: CCZ directly on all-ones.
    qc.append(ccz, [0, 1, 2])

    # Mark |000>: conjugate with X so CCZ fires on all-zeros.
    qc.x([0, 1, 2])
    qc.append(ccz, [0, 1, 2])
    qc.x([0, 1, 2])

    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean) for n qubits."""
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    ccz = ZGate().control(n - 1)
    qc.append(ccz, list(range(n)))
    qc.x(range(n))
    qc.h(range(n))
    return qc


def run_grover(n: int, targets: set, iterations: int) -> dict:
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    oracle = build_oracle(n, targets)
    diffuser = build_diffuser(n)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n))
        qc.append(diffuser.to_gate(), range(n))

    qc.measure(range(n), range(n))
    qc = qc.decompose(reps=3)

    sim = AerSimulator()
    result = sim.run(qc, shots=4096).result()
    return result.get_counts()


def main() -> bool:
    n = 3

    classical_targets = classical_monochromatic_colorings(n)
    assert classical_targets == {0, 7}, classical_targets

    # Optimal Grover iteration count for N=8, M=2: floor(pi/4 * sqrt(N/M)).
    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt((2 ** n) / len(classical_targets)))))

    counts = run_grover(n, classical_targets, iterations)

    # Convert measured bitstrings (Qiskit little-endian c-string) to ints.
    int_counts = {}
    for bitstring, c in counts.items():
        val = int(bitstring, 2)
        int_counts[val] = int_counts.get(val, 0) + c

    top2 = sorted(int_counts.items(), key=lambda kv: -kv[1])[:2]
    measured_top2 = {v for v, _ in top2}

    print("Erdos problem #118 (tags: set theory, ramsey theory; oeis: N/A)")
    print("Classical monochromatic-triangle colorings (brute force):", sorted(classical_targets))
    print("Grover iterations used:", iterations)
    print("Measured counts (as integers):", dict(sorted(int_counts.items())))
    print("Top-2 measured outcomes:", sorted(measured_top2))

    ok = measured_top2 == classical_targets
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
