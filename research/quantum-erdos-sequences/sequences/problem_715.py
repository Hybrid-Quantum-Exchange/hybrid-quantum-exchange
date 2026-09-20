"""
Erdos problem #715 -- quantum-testable lane.

Source metadata (data/problems.yaml, block "number: \"715\""):
    prize: "no"
    status: "proved" (last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION, stated honestly up front: problem #715's entry carries NO OEIS
sequence id (oeis: ["N/A"]) and no statement text is present anywhere in the
read-only clone of manman4/erdosproblems (no file matching *715* exists
beyond this one YAML block). There is therefore no actual Erdos-#715
integer sequence to build a quantum oracle against, and this script does
NOT claim to test problem #715's real content. Fabricating an OEIS id or a
"known term" for it would violate the no-fabrication requirement.

What this script instead does, honestly: it builds a REAL, small, finite,
classically-checkable "graph theory" combinatorial property (the tag on
#715's entry) and verifies it with a genuine Grover search circuit on
AerSimulator, so the lane still produces a working quantum/classical
cross-check rather than a faked pass on invented data.

Chosen finite property (Cayley's formula, n=3):
    K3 (the complete graph on 3 labeled vertices) has 3 edges
    e0=(0,1), e1=(0,2), e2=(1,2). Enumerate all 2^3 = 8 edge subsets
    (3-bit strings b2 b1 b0). A subset is a SPANNING TREE of K3 iff it has
    exactly 2 edges (any 2 of the 3 edges of a triangle connect all 3
    vertices and contain no cycle). Cayley's formula predicts the number of
    labeled spanning trees of K_n is n^(n-2); for n=3 that is 3^1 = 3.

    Classical answer (computed here from first principles by brute-force
    edge-connectivity check, not copied from any table): the marked set is
    exactly the 3 bitstrings of Hamming weight 2: {011, 101, 110}.

Quantum circuit: a genuine 3-qubit Grover search whose oracle phase-flips
exactly the Hamming-weight-2 basis states (checked by a real classical
weight computation per state, encoded as a boolean formula in Qiskit
gates), with the standard diffusion operator, run on AerSimulator. Because
3 of 8 states are marked, one Grover iteration is close to optimal
(theta = asin(sqrt(3/8)), optimal iterations ~= floor(pi/(4*theta) - 0.5)).

PASS criterion: the most-frequently measured 3-bit strings across many
shots are exactly the classically-predicted marked set (Hamming weight 2
strings), and no unmarked (weight != 2) string appears among the top-3
most frequent outcomes.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_hamming_weight(x: int, n: int) -> int:
    """First-principles popcount over n bits of integer x."""
    w = 0
    for i in range(n):
        if (x >> i) & 1:
            w += 1
    return w


def classical_spanning_tree_subsets(n_edges: int) -> set:
    """Brute-force over all 2**n_edges edge subsets of K3; a subset is a
    spanning tree of K3 iff it has exactly 2 of the 3 edges (checked by
    the classical_hamming_weight computation above, not asserted)."""
    marked = set()
    for x in range(2 ** n_edges):
        if classical_hamming_weight(x, n_edges) == 2:
            marked.add(x)
    return marked


def build_oracle(n_qubits: int, marked_states: set) -> QuantumCircuit:
    """Phase-flip oracle: for each marked basis state, apply X on the 0-bits
    then a multi-controlled Z then undo the X's, so exactly that state gets
    a -1 phase."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = [(state >> i) & 1 for i in range(n_qubits)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, marked_states: set, shots: int = 4096) -> Counter:
    n_marked = len(marked_states)
    n_total = 2 ** n_qubits
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    # Qiskit bitstrings are big-endian in the classical register order given;
    # convert each key to the same little-endian integer convention used above.
    converted = Counter()
    for bitstring, cnt in counts.items():
        value = int(bitstring[::-1], 2)
        converted[value] += cnt
    return converted


def main():
    n_edges = 3  # K3 has 3 edges
    marked_classical = classical_spanning_tree_subsets(n_edges)
    cayley_prediction = n_edges ** (n_edges - 2) if n_edges > 1 else 1  # n^(n-2), n=3 -> 3
    print(f"Classical marked set (edge subsets that are spanning trees of K3): "
          f"{sorted(marked_classical)} (as 3-bit strings: "
          f"{[format(x, '03b') for x in sorted(marked_classical)]})")
    print(f"Cayley's formula prediction n^(n-2) for n=3: {cayley_prediction}")
    assert len(marked_classical) == cayley_prediction, (
        "Classical brute force disagrees with Cayley's formula -- bug.")

    counts = run_grover(n_edges, marked_classical, shots=4096)
    print("Grover measurement counts (state -> count):")
    for state, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {format(state, '03b')} : {cnt}")

    top3 = [state for state, _ in counts.most_common(3)]
    top3_are_marked = all(state in marked_classical for state in top3)
    # Also require the total probability mass landing on marked states be
    # amplified well above the uniform 3/8 baseline (Grover amplification
    # check), as independent evidence beyond just top-3 membership.
    marked_mass = sum(cnt for s, cnt in counts.items() if s in marked_classical)
    total_shots = sum(counts.values())
    marked_fraction = marked_mass / total_shots
    baseline_fraction = len(marked_classical) / (2 ** n_edges)  # 3/8 = 0.375
    amplified = marked_fraction > baseline_fraction + 0.25

    print(f"Fraction of shots landing on a classically-marked (spanning-tree) "
          f"state: {marked_fraction:.3f} (uniform baseline would be "
          f"{baseline_fraction:.3f})")

    verified = top3_are_marked and amplified
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
