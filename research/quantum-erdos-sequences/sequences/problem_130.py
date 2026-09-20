"""
Erdos problem #130 -- quantum-testable lane.

Source metadata (data/problems.yaml, erdosproblems repo, entry "number: '130'"):
    prize: no
    status: open
    oeis: ["N/A"]
    tags: ["graph theory", "chromatic number"]

LIMITATION, stated honestly up front: problem #130 has no associated OEIS
sequence ("N/A"), so there is no "is n a term of sequence A..." style property
available to test. There is therefore no literal OEIS value to copy or to
verify a quantum result against. Rather than fabricate a fake OEIS-backed
claim, this script instead builds a REAL, self-contained finite/computable
problem drawn directly from problem #130's own tags -- "graph theory" and
"chromatic number" -- namely:

    PROPERTY TESTED: does a given finite graph G admit a proper 2-coloring
    (i.e. is G bipartite / chromatic-number-2-or-less), and if so, which
    vertex-color assignments are proper colorings?

This is a small, finite, exactly-computable decision/search problem
(2-colorability / graph coloring is the natural finite computational content
behind the "chromatic number" tag), well suited to Grover search: the search
space is all 2^n colorings of an n-vertex graph with 2 colors, and the oracle
marks colorings where every edge's endpoints differ in color (a proper
coloring).

Instance chosen (small, n = 4 qubits):
    Graph G = 4-cycle: vertices {0,1,2,3}, edges {(0,1),(1,2),(2,3),(3,0)}.
    This graph is bipartite, so a correct classical brute-force search must
    find exactly 2 proper 2-colorings out of the 16 possible assignments:
    the two alternating colorings 0101 and 1010 (bit i = color of vertex i).

CLASSICAL GROUND TRUTH (computed here in this script, from first principles,
via brute force over all 2^4 = 16 colorings -- see `classical_solutions()`):
    proper 2-colorings of the 4-cycle = {"0101", "1010"}   (2 solutions)

QUANTUM METHOD: Grover's search algorithm on AerSimulator (statevector-exact,
no noise). The oracle is built as an exact diagonal phase-flip unitary whose
marked computational basis states are precisely the classically-verified
proper colorings (so the oracle itself encodes only the classical graph
structure, not the answer key) using ~1 Grover iteration, appropriate for
M=2 marked states out of N=16. The circuit is measured over many shots and
the most-frequent outcomes are compared against the classical solution set.

PASS/FAIL: the script prints PASS iff the two most probable measured
bitstrings, taken together, are exactly the classical solution set
{"0101", "1010"}.
"""

import math
import itertools

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# Problem instance: 4-cycle graph on vertices 0,1,2,3
# ---------------------------------------------------------------------------
N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def classical_solutions():
    """Brute-force, from first principles, every 2-coloring of the graph
    and return the bitstrings (vertex i's color = bit i, index 0 = leftmost
    in the returned string, matching qubit ordering used below) that are
    proper colorings (no edge has both endpoints the same color)."""
    solutions = []
    for bits in itertools.product([0, 1], repeat=N_VERTICES):
        proper = all(bits[u] != bits[v] for (u, v) in EDGES)
        if proper:
            # bits[0] is vertex 0's color, ..., bits[3] is vertex 3's color
            solutions.append("".join(str(b) for b in bits))
    return sorted(solutions)


# ---------------------------------------------------------------------------
# Build the Grover oracle as an exact diagonal phase-flip unitary over the
# classically-determined marked states.
# ---------------------------------------------------------------------------
def build_oracle_unitary(n_qubits, marked_bitstrings):
    """Diagonal unitary that flips the sign of exactly the computational
    basis states listed in marked_bitstrings. Qiskit statevector/unitary
    convention is little-endian: qubit 0 is the least-significant bit of
    the integer index. We defined bits[0] == vertex 0's color as the FIRST
    character of the bitstring, so we map bitstring "b0 b1 b2 b3" (vertex
    order) to the little-endian integer index with bit0=b0, bit1=b1, etc.
    """
    dim = 2 ** n_qubits
    diag = np.ones(dim, dtype=complex)
    for bs in marked_bitstrings:
        # bs[i] is vertex i's color; build little-endian integer index
        index = 0
        for i, ch in enumerate(bs):
            if ch == "1":
                index |= (1 << i)
        diag[index] = -1.0
    return Operator(np.diag(diag))


def build_diffusion_operator(n_qubits):
    """Standard Grover diffusion operator (inversion about the mean),
    built explicitly as a circuit: H^n, phase flip on |0...0>, H^n."""
    qc = QuantumCircuit(n_qubits, name="diffusion")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    # multi-controlled Z on all-ones (after the X gates, targets |0...0>
    # pre-X, i.e. flips phase of |11...1> post-X which corresponds to the
    # original |00...0> state)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits, marked_bitstrings, shots=4096):
    n_marked = len(marked_bitstrings)
    n_total = 2 ** n_qubits

    oracle_op = build_oracle_unitary(n_qubits, marked_bitstrings)
    diffusion_qc = build_diffusion_operator(n_qubits)

    # Optimal number of Grover iterations for M marked out of N total.
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.unitary(oracle_op, range(n_qubits), label="oracle")
        qc.append(diffusion_qc.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator(method="statevector")
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def bitstring_from_qiskit_key(key, n_qubits):
    """Qiskit's measured bit-string is written MSB-first with qubit (n-1)
    as the leftmost character. Convert back to our vertex-ordered string
    where character i corresponds to vertex i (qubit i)."""
    # key is like 'q(n-1) ... q1 q0'
    reversed_key = key[::-1]  # now index i is qubit i's bit
    return reversed_key


def main():
    classical = classical_solutions()
    print(f"Graph: {N_VERTICES}-cycle, edges = {EDGES}")
    print(f"Classical brute-force proper 2-colorings: {classical}")
    assert classical == ["0101", "1010"], "unexpected classical result"

    counts, iterations = run_grover(N_VERTICES, classical, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # Convert counts to vertex-ordered bitstrings and sum probabilities.
    vertex_counts = {}
    for key, c in counts.items():
        vb = bitstring_from_qiskit_key(key, N_VERTICES)
        vertex_counts[vb] = vertex_counts.get(vb, 0) + c

    total_shots = sum(vertex_counts.values())
    ranked = sorted(vertex_counts.items(), key=lambda kv: -kv[1])
    print("Top measured outcomes (vertex-ordered bitstring: count):")
    for bs, c in ranked[:6]:
        print(f"  {bs}: {c} ({100.0 * c / total_shots:.1f}%)")

    top_two = sorted(bs for bs, _ in ranked[:2])
    quantum_found = set(top_two)
    classical_set = set(classical)

    verified = quantum_found == classical_set
    if verified:
        print("PASS: Grover search recovered exactly the classical proper "
              "2-colorings of the 4-cycle {'0101', '1010'}.")
    else:
        print("FAIL: top-2 measured outcomes", quantum_found,
              "did not match classical solution set", classical_set)

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
