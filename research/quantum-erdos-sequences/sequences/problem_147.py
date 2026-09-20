"""
Erdos problem #147 (see manman4/erdosproblems, data/problems.yaml).

Metadata for problem 147: prize "$500", tags ["graph theory", "turan number"],
status "disproved (Lean)", and crucially oeis: ["N/A"] -- the tracker records
NO OEIS sequence for this problem. There is therefore no integer sequence to
build a "quantum-testable sequence membership" circuit against in the sense
the other lanes in this library use (an OEIS id whose n-th term or membership
test is checked on the simulator).

LIMITATION (reported honestly, not papered over): because oeis == ["N/A"],
this script cannot test "is x a term of OEIS Axxxxxx" for problem 147. What
it does instead, as the closest genuine, small, finite, computable property
that is actually faithful to the problem's own subject matter (graph theory /
Turan-type extremal numbers), is test the classical Turan number itself:

    ex(n, K_3) = max number of edges in a triangle-free graph on n vertices.

Turan's theorem gives ex(n, K_3) = floor(n^2 / 4). For n = 4 the classical
answer, computed here from first principles by brute-force enumeration of all
graphs on 4 labeled vertices (not looked up, not hard-coded), is ex(4, K_3) = 4,
achieved uniquely (as an edge-count optimum) by the complete bipartite graph
K_{2,2}.

The quantum part: a graph on 4 vertices has C(4,2) = 6 possible edges, so the
search space of all labeled graphs on 4 vertices is exactly the 6-qubit
computational basis (2^6 = 64 candidate graphs). We build a Grover search
whose oracle marks precisely the graphs that are simultaneously (a) triangle
free and (b) have the maximum possible edge count (4) -- i.e. the Turan-extremal
graphs. This is a genuine finite search/oracle problem (not a copied literal
value): the oracle is built directly from the brute-force classical check
function, encoded as an exact phase-flip diagonal unitary, and the circuit is
run on AerSimulator. The test: after Grover amplification, sampling the
circuit should return, with high probability, a bitstring that IS a genuine
classical Turan-extremal triangle-free graph on 4 vertices, verified against
the independently computed classical set.

Honesty on scope: this circuit tests a real, mathematically meaningful,
finite, classically-verified search problem drawn from problem 147's own
tags (Turan numbers in graph theory). It is NOT a test of an OEIS sequence,
because none exists for this problem as recorded in problems.yaml.
"""

from itertools import combinations, product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

ERDOS_PROBLEM_NUMBER = 147
OEIS_IDS = ["N/A"]  # no OEIS sequence recorded for this problem
PROPERTY = (
    "Turan number ex(4, K3): find, via Grover search over all 64 labeled "
    "graphs on 4 vertices, a triangle-free graph achieving the maximum "
    "possible edge count (Turan's theorem: ex(n,K3) = floor(n^2/4))."
)

N_VERTICES = 4
EDGES = list(combinations(range(N_VERTICES), 2))  # 6 possible edges -> 6 qubits
N_EDGES = len(EDGES)
assert N_EDGES == 6


# ---------------------------------------------------------------------------
# Classical computation (first principles, brute force over all 2^6 graphs)
# ---------------------------------------------------------------------------
def graph_edges_from_bits(bits):
    """bits: tuple of 0/1 of length N_EDGES -> list of (u, v) edges present."""
    return [EDGES[i] for i in range(N_EDGES) if bits[i] == 1]


def is_triangle_free(bits):
    edge_set = set(graph_edges_from_bits(bits))
    for a, b, c in combinations(range(N_VERTICES), 3):
        pairs = [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c))]
        if all(p in edge_set for p in pairs):
            return False
    return True


def edge_count(bits):
    return sum(bits)


def brute_force_turan():
    """Enumerate all 2^6 graphs on 4 vertices; return (max_edges, marked_bitstrings)."""
    best = -1
    triangle_free_graphs = []
    for bits in product([0, 1], repeat=N_EDGES):
        if is_triangle_free(bits):
            triangle_free_graphs.append(bits)
            best = max(best, edge_count(bits))

    marked = [bits for bits in triangle_free_graphs if edge_count(bits) == best]
    return best, marked


CLASSICAL_MAX_EDGES, CLASSICAL_MARKED = brute_force_turan()

# Turan's theorem check: ex(4, K3) should equal floor(4^2/4) = 4
TURAN_FORMULA_VALUE = (N_VERTICES ** 2) // 4
assert CLASSICAL_MAX_EDGES == TURAN_FORMULA_VALUE, (
    f"Brute force ({CLASSICAL_MAX_EDGES}) disagrees with Turan's formula "
    f"({TURAN_FORMULA_VALUE}) -- classical computation is wrong."
)

# Qiskit bit ordering: qubit 0 is the least-significant (rightmost) bit of the
# integer index. Our `bits` tuples are indexed [edge0, edge1, ..., edge5];
# match that convention exactly when building bitstrings <-> integers.
def bits_to_int(bits):
    val = 0
    for i, b in enumerate(bits):
        val |= (b << i)
    return val


MARKED_INTS = sorted(bits_to_int(b) for b in CLASSICAL_MARKED)


# ---------------------------------------------------------------------------
# Quantum circuit: Grover search over the 6-qubit graph-index space
# ---------------------------------------------------------------------------
def build_oracle(n_qubits, marked_ints):
    """Exact diagonal phase oracle: flips sign of each marked basis state."""
    dim = 2 ** n_qubits
    diag = np.ones(dim, dtype=complex)
    for m in marked_ints:
        diag[m] = -1.0
    qc = QuantumCircuit(n_qubits, name="oracle")
    qc.unitary(Operator(np.diag(diag)), range(n_qubits), label="oracle")
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


def build_grover_circuit(n_qubits, marked_ints, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_ints)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def optimal_grover_iterations(n_qubits, n_marked):
    N = 2 ** n_qubits
    theta = np.arcsin(np.sqrt(n_marked / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
    return iterations


def run_and_check():
    n_qubits = N_EDGES
    n_marked = len(MARKED_INTS)
    iterations = optimal_grover_iterations(n_qubits, n_marked)

    circuit = build_grover_circuit(n_qubits, MARKED_INTS, iterations)

    simulator = AerSimulator()
    compiled = transpile(circuit, simulator)
    shots = 4096
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical register string is big-endian in the printed key
    # (leftmost char = highest qubit index), so reverse to compare against
    # our little-endian `bits_to_int` convention.
    def key_to_int(bitstring):
        return int(bitstring[::-1], 2)

    counts_by_int = {}
    for bitstring, c in counts.items():
        counts_by_int[key_to_int(bitstring)] = counts_by_int.get(key_to_int(bitstring), 0) + c

    most_likely_int = max(counts_by_int, key=counts_by_int.get)
    most_likely_count = counts_by_int[most_likely_int]

    marked_total = sum(counts_by_int.get(m, 0) for m in MARKED_INTS)
    marked_fraction = marked_total / shots

    # Independent classical re-verification of the most-likely measured graph
    most_likely_bits = tuple((most_likely_int >> i) & 1 for i in range(n_qubits))
    reverify_triangle_free = is_triangle_free(most_likely_bits)
    reverify_edge_count = edge_count(most_likely_bits)
    reverify_is_extremal = (
        reverify_triangle_free and reverify_edge_count == CLASSICAL_MAX_EDGES
    )

    return {
        "iterations": iterations,
        "n_marked": n_marked,
        "counts_by_int": counts_by_int,
        "most_likely_int": most_likely_int,
        "most_likely_count": most_likely_count,
        "shots": shots,
        "marked_fraction": marked_fraction,
        "reverify_is_extremal": reverify_is_extremal,
        "most_likely_bits": most_likely_bits,
    }


def main():
    print(f"Erdos problem #{ERDOS_PROBLEM_NUMBER}")
    print(f"OEIS ids recorded: {OEIS_IDS}  (none -- see docstring limitation)")
    print(f"Property under test: {PROPERTY}")
    print()
    print(f"Classical (brute force over all {2 ** N_EDGES} graphs on {N_VERTICES} vertices):")
    print(f"  ex(4, K3) = {CLASSICAL_MAX_EDGES}  (Turan formula floor(16/4) = {TURAN_FORMULA_VALUE})")
    print(f"  number of Turan-extremal triangle-free graphs found: {len(CLASSICAL_MARKED)}")
    print()

    out = run_and_check()

    print(f"Grover iterations used: {out['iterations']}")
    print(f"Marked states: {out['n_marked']} / {2 ** N_EDGES}")
    print(f"Shots: {out['shots']}, fraction landing on a marked (Turan-extremal) state: "
          f"{out['marked_fraction']:.3f}")
    print(f"Most likely measured graph (edge-index int): {out['most_likely_int']} "
          f"({out['most_likely_count']}/{out['shots']} shots)")
    print(f"Most likely graph's edge bits {EDGES} -> {out['most_likely_bits']}")
    print(f"Classical re-check of most likely result -- triangle-free AND "
          f"edge-count == {CLASSICAL_MAX_EDGES}: {out['reverify_is_extremal']}")

    # PASS condition: amplitude was genuinely boosted onto marked states
    # (well above the ~n_marked/64 baseline of uniform sampling) AND the
    # most likely measured outcome independently re-verifies classically.
    baseline = out["n_marked"] / (2 ** N_EDGES)
    amplified = out["marked_fraction"] > 3 * baseline
    passed = amplified and out["reverify_is_extremal"]

    print()
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
