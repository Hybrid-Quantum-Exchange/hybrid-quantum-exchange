"""
Erdos problem #577 -- quantum-testable lane (best-effort, with an honest limitation).

Source metadata (from erdosproblems.com's data file, problems.yaml, entry
"number: '577'"):
    prize: no
    status: proved (last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION (read before trusting anything below as "problem 577"):
Problem #577 has NO associated OEIS sequence id in the source data (the oeis
field is literally ["N/A"]). The task that generated this file requires
identifying a property of an OEIS sequence tied to the problem; there is none
to identify here, so nothing in this script can honestly be described as a
"quantum-testable sequence" derived from problem 577's actual content. The
repository clone available to this script does not contain the full text of
problem 577 (only the YAML metadata above), so the specific combinatorial
statement of the problem is also unknown here.

Rather than fabricate an OEIS-derived property or invent problem content this
script has no access to, this script instead builds a genuine, small,
finite, computable decision problem from the one real piece of information
available -- the tag "graph theory" -- and tests it with a real Grover search
circuit on Qiskit's ideal AerSimulator. This is NOT a verified restatement of
problem 577 and should not be reported as one; it is the best honest
substitute allowed by the task instructions when "no OEIS id" applies.

The chosen finite, computable graph-theory property:
    Fix the 4-vertex graph G with vertex set {0,1,2,3} and edge set
        E = {(0,1), (1,2), (2,0), (0,3)}
    (a triangle 0-1-2 plus a pendant edge to vertex 3).
    Classical question: does G contain a triangle (3 mutually adjacent
    vertices)? Search space: all C(4,3) = 4 unordered triples of vertices,
    indexed by a 2-qubit register (with one dummy index padded out to the
    next power of two, i.e. 4 = 2^2 exactly).

    This is computed from first principles classically in `classical_answer()`
    below (no OEIS lookup, no hard-coded literal answer), and then verified
    with a Grover search circuit whose oracle marks exactly the triangle
    index computed classically.

Circuit: standard Grover's algorithm, 2 index qubits (4 items), oracle built
by a classically-computed marked index (converted to a multi-controlled Z
via X-gates + MCZ), one diffusion operator (optimal for 1 marked item out of
4, i.e. ceil(pi/4 * sqrt(N/M)) = 1 iteration), run on qiskit_aer's
AerSimulator with statevector-exact simulation (shots), then compared to the
classical answer.
"""

import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = {(0, 1), (1, 2), (2, 0), (0, 3)}


def is_edge(u, v):
    return (u, v) in EDGES or (v, u) in EDGES


def all_triples():
    """All C(4,3) = 4 unordered vertex triples, in a fixed deterministic order."""
    return list(itertools.combinations(VERTICES, 3))


def classical_answer():
    """
    Returns (marked_index, triple) where `triple` is the unique triangle
    among the 4 vertex-triples of G (computed by brute force, no lookup),
    and `marked_index` is its position (0..3) in all_triples().

    Raises if the number of triangles found is not exactly 1, since the
    Grover oracle built below assumes a single marked item.
    """
    triples = all_triples()
    triangle_indices = [
        i
        for i, (a, b, c) in enumerate(triples)
        if is_edge(a, b) and is_edge(b, c) and is_edge(a, c)
    ]
    assert len(triangle_indices) == 1, (
        f"expected exactly one triangle in G, found {len(triangle_indices)}: "
        f"{[triples[i] for i in triangle_indices]}"
    )
    idx = triangle_indices[0]
    return idx, triples[idx]


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 4 triples (2 index qubits), oracle marks
#    exactly the classically-computed triangle index.
# ---------------------------------------------------------------------------

def build_oracle(marked_index, n_qubits):
    """Phase-flip oracle: multi-controlled Z on |marked_index>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked_index, f"0{n_qubits}b")
    # Flip qubits that should be 0 in the target index, so a normal MCZ
    # (controls on 1) fires exactly on |marked_index>.
    for i, b in enumerate(reversed(bits)):
        if b == "0":
            qc.x(i)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for i, b in enumerate(reversed(bits)):
        if b == "0":
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
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked_index, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked_index, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, range(n_qubits), inplace=True)
        qc.compose(diffuser, range(n_qubits), inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run_grover(marked_index, n_items=4, shots=2048):
    n_qubits = int(np.ceil(np.log2(n_items)))
    # Optimal iteration count for 1 marked item out of N=4: floor(pi/4*sqrt(4/1)) = 1
    # (this case is exact: 1 iteration drives the marked amplitude to 1).
    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(n_items / 1))))
    qc = build_grover_circuit(marked_index, n_qubits, iterations)

    sim = AerSimulator(method="statevector")
    job = sim.run(qc, shots=shots)
    counts = job.result().get_counts()

    best_bitstring = max(counts, key=counts.get)
    found_index = int(best_bitstring, 2)
    return found_index, counts


# ---------------------------------------------------------------------------
# 3. Verify quantum result against classical ground truth.
# ---------------------------------------------------------------------------

def main():
    classical_idx, classical_triple = classical_answer()
    print(f"Classical answer: triangle at index {classical_idx} -> vertices {classical_triple}")

    found_idx, counts = run_grover(classical_idx, n_items=len(all_triples()), shots=2048)
    print(f"Grover search result (most frequent measured index): {found_idx}")
    print(f"Measurement counts: {counts}")

    top_count = counts[format(found_idx, "02b")]
    total = sum(counts.values())
    print(f"Success probability (fraction of shots on marked index): {top_count / total:.3f}")

    ok = (found_idx == classical_idx) and (top_count / total > 0.5)

    print()
    print("NOTE: this verifies a Grover search over a small graph-theory decision")
    print("problem chosen because problem 577 has no associated OEIS sequence in")
    print("the source data (oeis: ['N/A']); it is a best-effort substitute, not a")
    print("verified restatement of problem 577's actual mathematical content.")
    print()
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
