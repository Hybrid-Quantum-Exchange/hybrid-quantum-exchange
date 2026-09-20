"""
Erdos problem #832 (from https://github.com/manman4/erdosproblems,
data/problems.yaml entry `number: "832"`) is about the chromatic number of
hypergraphs (tags: "graph theory", "hypergraphs", "chromatic number"). Its
YAML entry carries no real OEIS sequence id -- the field is
`oeis: ["possible"]`, a placeholder string, not an actual A-number. So this
script cannot test "membership in OEIS sequence A......" as instructed for
problems that do have one. Reported honestly: no OEIS id is used here.

LIMITATION: because there is no OEIS sequence to key off, this script instead
tests a small, finite, genuinely computable property that is faithful to the
problem's actual subject matter (hypergraph 2-colorability / property B,
which is exactly what "chromatic number of a hypergraph" is about):

    Classical property tested
    --------------------------
    Fix the 3-uniform hypergraph H on vertex set {0,1,2,3} with hyperedges

        e1 = {0,1,2}
        e2 = {0,1,3}
        e3 = {0,2,3}
        e4 = {1,2,3}

    (every 3-subset of a 4-set -- this is K4^(3), the complete 3-uniform
    hypergraph on 4 vertices). A 2-coloring c: {0,1,2,3} -> {0,1} is
    "proper" (witnesses 2-colorability / Property B) iff no hyperedge is
    monochromatic, i.e. for every hyperedge e, the colors of its vertices
    are not all equal.

    The classical answer (computed in this script, by brute force over all
    2^4 = 16 colorings, from first principles -- no external data) is the
    exact set of proper colorings. Since every hyperedge here is a 3-subset
    of the 4 vertices, a coloring is monochromatic on some hyperedge exactly
    when 3 or more of the 4 vertices share a color; equivalently, a coloring
    is proper exactly when it is a balanced 2-2 split (every 3-subset then
    contains vertices of both colors). Brute force confirms there are
    exactly 6 such balanced colorings (choose which 2 of 4 vertices get
    color 0: C(4,2) = 6), and no unbalanced (3-1 or 4-0) coloring is proper.
    This exhibits K4^(3)'s 2-colorability (property B) by direct
    enumeration.

    Quantum circuit
    ----------------
    4 qubits represent the coloring of vertices 0..3 (|0>=color0, |1>=color1).
    A Grover search is built whose oracle marks (phase-flips) exactly the
    classically-computed proper-coloring bitstrings, followed by the standard
    Grover diffusion operator, run on the ideal AerSimulator (statevector
    method). After the optimal number of Grover iterations the two marked
    basis states 0110 and 1001 should dominate the measured distribution.

    PASS/FAIL: the script computes the classical solution set by brute force,
    runs the Grover circuit, and checks that the two most frequent measured
    bitstrings (by counts) are exactly the classical solution set.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate

N_VERTICES = 4
HYPEREDGES = [
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
]


def is_proper_coloring(coloring):
    """coloring: tuple of 0/1 of length N_VERTICES. True iff no hyperedge
    is monochromatic under this coloring."""
    for e in HYPEREDGES:
        colors = {coloring[v] for v in e}
        if len(colors) == 1:
            return False
    return True


def classical_solution_set():
    """Brute force over all 2^N_VERTICES colorings; return the set of
    proper colorings as bitstrings (qubit 0 = vertex 0 = least significant
    bit, matching Qiskit's little-endian bit ordering)."""
    solutions = set()
    for bits in itertools.product((0, 1), repeat=N_VERTICES):
        # bits[i] is the color of vertex i
        if is_proper_coloring(bits):
            # Qiskit bitstrings print with qubit N-1 first (most significant)
            # ... qubit 0 last. Build the string accordingly.
            bitstring = "".join(str(bits[v]) for v in reversed(range(N_VERTICES)))
            solutions.add(bitstring)
    return solutions


def build_oracle(n_qubits, marked_bitstrings):
    """Phase-flip oracle: for each marked bitstring, apply X gates to map
    it to |11...1>, apply a multi-controlled Z (via H + MCX + H on the
    target), then undo the X gates."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bitstring in marked_bitstrings:
        # bitstring[0] corresponds to qubit n_qubits-1 (Qiskit's printed
        # order), bitstring[-1] corresponds to qubit 0.
        bits = [int(b) for b in reversed(bitstring)]  # bits[i] -> qubit i
        zero_qubits = [i for i, b in enumerate(bits) if b == 0]
        for q in zero_qubits:
            qc.x(q)
        # multi-controlled Z on all n_qubits: use qubit n_qubits-1 as target
        # sandwiched in H gates, controlled on the rest.
        target = n_qubits - 1
        controls = [i for i in range(n_qubits) if i != target]
        qc.h(target)
        qc.append(MCXGate(len(controls)), controls + [target])
        qc.h(target)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    target = n_qubits - 1
    controls = [i for i in range(n_qubits) if i != target]
    qc.h(target)
    qc.append(MCXGate(len(controls)), controls + [target])
    qc.h(target)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits, marked_bitstrings, shots=4096):
    n_marked = len(marked_bitstrings)
    n_total = 2 ** n_qubits
    # optimal number of Grover iterations
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_bitstrings)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator(method="statevector")
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    classical_solutions = classical_solution_set()
    print("Classical proper 2-colorings of K4^(3) (bitstrings):", sorted(classical_solutions))

    counts, iterations = run_grover(N_VERTICES, classical_solutions)
    print(f"Grover iterations used: {iterations}")
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print("Top measured outcomes:", sorted_counts[:6])

    top_k = {b for b, _ in sorted_counts[: len(classical_solutions)]}

    verified = top_k == classical_solutions
    if verified:
        print("PASS: Grover search's top outcomes match the classically "
              "computed proper 2-colorings of K4^(3).")
    else:
        print("FAIL: quantum result does not match classical computation.")
        print("Expected:", classical_solutions, "Got top-k:", top_k)

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
