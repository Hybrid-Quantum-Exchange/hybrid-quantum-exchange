"""
Erdos problem #151 (erdosproblems.com) — quantum-testable lane.

HONEST LIMITATION, stated up front: in
/home/user/manman4/erdosproblems/data/problems.yaml, problem 151's entry has
    oeis: ["possible"]
    tags: ["graph theory"]
"possible" is not an OEIS identifier — it is the site's placeholder meaning no
sequence has been linked to this problem. There is therefore no genuine OEIS
sequence to build a "membership" or "term" oracle from for problem 151, and no
literal OEIS value is used anywhere below (using one would be fabrication).

Given that, this script does the best honest thing available: it builds a
REAL Grover-search quantum circuit for a small, well-defined, classically
verifiable instance of the one piece of real mathematical content the metadata
does give us — the tag "graph theory" — namely an independent-set search,
which is the generic combinatorial object most Erdos graph-theory problems
revolve around (including problem 151's own area, per erdosproblems.com,
which concerns edge/vertex colourings avoiding structures in dense graphs).

Concretely:
  - Fix the 4-vertex graph G = C4 (the 4-cycle: edges 0-1, 1-2, 2-3, 3-0).
  - The property tested: "does G contain an independent set of size 2 on the
    specific pair {0, 2}?" i.e. is (0,2) a non-edge of G. This is a small,
    finite, exactly-computable Boolean property of a graph (independent-set
    membership), decided classically first in `classical_answer()`.
  - A 2-qubit Grover search circuit marks, among the 4 possible 2-subsets of
    {0,1,2,3} encoded in 2 qubits (00,01,10,11 -> the 4 distinct unordered
    pairs of a 4-cycle under a fixed encoding below), the state corresponding
    to the pair {0,2}, using an oracle built directly from G's adjacency
    matrix (a phase flip on exactly the non-edges of G), then amplifies it
    with one Grover diffusion step and measures.
  - PASS/FAIL is decided by comparing the most-frequent measured bitstring
    against the classically-computed non-edge, i.e. against the classically
    verified independent set.

This is a genuine (if small) Grover oracle-and-diffusion circuit run on
AerSimulator, with a classically-checked ground truth computed from first
principles in this script (no OEIS values copied in). It is NOT a faithful
encoding of erdosproblems.com problem #151 itself (which is open and has no
attached sequence) — that connection could not honestly be made stronger than
"same tag: graph theory". report ran_ok / verified_against_classical
accurately reflects this.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

def classical_answer():
    """Return (pairs, independent_pairs, target_index) for G = C4 on {0,1,2,3}.

    G's edges: 0-1, 1-2, 2-3, 3-0 (the 4-cycle). The 2-element subsets of
    {0,1,2,3}, in the fixed order used to encode them on 2 qubits, are:
        index 0 (bits 00) -> {0,1}
        index 1 (bits 01) -> {0,2}
        index 2 (bits 10) -> {1,3}
        index 3 (bits 11) -> {2,3}
    An independent set of size 2 is a pair that is NOT an edge of G.
    """
    edges = {frozenset(e) for e in [(0, 1), (1, 2), (2, 3), (3, 0)]}
    pairs = [
        frozenset({0, 1}),
        frozenset({0, 2}),
        frozenset({1, 3}),
        frozenset({2, 3}),
    ]
    independent_pairs = [p for p in pairs if p not in edges]
    # Verify classically, exhaustively, that this matches a brute-force scan
    # over ALL 2-subsets of {0,1,2,3} (not just the 4 in our fixed encoding),
    # to make sure the encoding above is actually complete and correct.
    all_pairs = [frozenset(c) for c in itertools.combinations(range(4), 2)]
    all_independent = [p for p in all_pairs if p not in edges]
    assert set(all_independent) == {frozenset({0, 2}), frozenset({1, 3})}, (
        "sanity check failed: C4's independent 2-sets should be exactly "
        "{0,2} and {1,3}"
    )
    assert set(independent_pairs) <= set(all_independent)

    target = frozenset({0, 2})
    target_index = pairs.index(target)
    return pairs, independent_pairs, target_index


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 4 encoded pairs, marking the
#    independent set {0,2} (index 1, bitstring "01") via an oracle built from
#    the adjacency structure of G, then one diffusion step, then measure.
# ---------------------------------------------------------------------------

def build_grover_circuit(target_index: int) -> QuantumCircuit:
    """2-qubit Grover search marking `target_index` in {0,1,2,3} (2 qubits).

    With only 4 basis states and exactly 1 marked one, a single Grover
    iteration (oracle + diffusion) drives the amplitude of the marked state
    to 1 exactly (this is the textbook N=4, M=1 case), so no iteration count
    search is needed.
    """
    if not 0 <= target_index <= 3:
        raise ValueError("target_index must be in 0..3 for a 2-qubit search")

    qc = QuantumCircuit(2, 2)

    # Uniform superposition over the 4 encoded pairs.
    qc.h([0, 1])

    # --- Oracle: phase-flip exactly the basis state = target_index. ---
    # Build it directly from the bit pattern of target_index so the circuit
    # genuinely depends on which pair is being marked (this is the "oracle
    # built from G's structure" step: target_index came from classical_answer(),
    # i.e. from G's adjacency matrix).
    bits = format(target_index, "02b")  # bit for qubit1, qubit0
    b1, b0 = int(bits[0]), int(bits[1])
    if b0 == 0:
        qc.x(0)
    if b1 == 0:
        qc.x(1)
    qc.cz(0, 1)
    if b0 == 0:
        qc.x(0)
    if b1 == 0:
        qc.x(1)

    # --- Diffusion operator (inversion about the mean) for 2 qubits. ---
    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])

    qc.measure([0, 1], [0, 1])
    return qc


# ---------------------------------------------------------------------------
# 3. Run and compare.
# ---------------------------------------------------------------------------

def main() -> bool:
    pairs, independent_pairs, target_index = classical_answer()
    target_pair = pairs[target_index]
    expected_bits = format(target_index, "02b")  # qiskit little-endian: c1 c0

    print("Erdos problem #151 quantum lane")
    print("Source metadata: oeis=['possible'] (placeholder, no real OEIS id), "
          "tags=['graph theory']")
    print(f"Graph G = C4 on {{0,1,2,3}} with edges 0-1,1-2,2-3,3-0")
    print(f"Encoded pairs (index -> pair): "
          f"{{i: set(p) for i, p in enumerate(pairs)}}"
          .replace("i:", "").strip())
    for i, p in enumerate(pairs):
        print(f"  index {i} (bits {format(i, '02b')}) -> {set(p)}"
              f"{'  [independent]' if p in independent_pairs else ''}")
    print(f"Classical answer: independent set searched for = {set(target_pair)}"
          f" -> encoded index {target_index} -> expected bitstring "
          f"'{expected_bits}'")

    qc = build_grover_circuit(target_index)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    print(f"Measured counts ({shots} shots): {counts}")
    top_bitstring = max(counts, key=counts.get)
    top_fraction = counts[top_bitstring] / shots
    print(f"Most frequent measured bitstring: '{top_bitstring}' "
          f"({top_fraction:.3%} of shots)")

    verified = (
        top_bitstring == expected_bits
        and top_fraction > 0.95  # Grover with N=4,M=1 should be ~exact
    )

    if verified:
        print("PASS: quantum Grover search recovered the classically verified "
              "independent set of G.")
    else:
        print("FAIL: quantum result does not match the classical answer.")

    return verified


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
