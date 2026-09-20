"""
Erdos problem #580 (https://www.erdosproblems.com/580) -- quantum-testable lane.

Source lookup (/home/user/manman4/erdosproblems/data/problems.yaml, block
"number: \"580\""): prize "no", status "decidable", tags ["graph theory"],
oeis: ["N/A"]. There is no OEIS sequence id attached to problem #580 and no
local problem-statement text is available in the cloned repo beyond the
README index row, so the specific extremal/counting quantity Erdos posed in
#580 cannot be recovered here. Per the task's fallback instructions ("If
after reasonable effort no genuine quantum circuit can be constructed for
this problem's sequence -- no OEIS id ... -- write the script anyway with
your best honest attempt, note the limitation clearly"), this script does
NOT claim to test problem #580's actual statement. Instead it builds a real,
verifiable Grover-search circuit for a small, finite, computable decision
property from the same tag ("graph theory") that #580 carries: which of the
64 labeled graphs on 4 vertices contain at least one triangle.

Classical property tested
--------------------------
Fix the 4 vertices {0,1,2,3} and their 6 possible edges, indexed 0..5:
  edge 0: (0,1)   edge 1: (0,2)   edge 2: (0,3)
  edge 3: (1,2)   edge 4: (1,3)   edge 5: (2,3)
A graph on these 4 vertices is encoded as a 6-bit integer n in [0, 64), bit i
of n telling whether edge i is present. There are 4 possible triangles:
  {0,1,2} -> edges {0,1,3}
  {0,1,3} -> edges {0,2,4}
  {0,2,3} -> edges {1,2,5}
  {1,2,3} -> edges {3,4,5}
A graph n is MARKED if it contains at least one complete triangle (all three
of some triangle's edges present). This is exactly computed classically in
`classical_marked_set()` below by brute force over all 64 graphs (n <= 64,
so the instance is small and the search space is the full N = 64 required by
the task). The classical answer -- the exact set and count of triangle-
containing graphs among the 64 -- is computed from first principles (no
literature value copied) and printed.

Quantum method
---------------
Grover's algorithm on 6 qubits (search space size N = 64). The oracle is a
diagonal +1/-1 unitary built directly from the classically-computed marked
set (so the circuit's oracle and the classical check are provably the same
predicate). The optimal number of Grover iterations for M marked items out
of N is computed as round((pi/4) * sqrt(N/M)) - 1 (0-indexed formula used
here), and the AerSimulator (ideal, no noise) is run with many shots. PASS
requires that Grover search amplifies measurement probability onto the
classically-marked triangle-containing graphs to a large majority of shots,
i.e. the circuit's dominant measured outcomes are exactly the classically
verified triangle-containing graphs.

Limitation: this substitutes a same-tag, small, finite, honestly-derived
graph property for problem #580's own (unrecoverable-from-source) statement.
It is not a claim about #580's specific open/solved content.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator

NUM_VERTICES = 4
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def triangle_edge_bits(tri):
    a, b, c = tri
    pairs = [(a, b), (a, c), (b, c)]
    return [EDGE_INDEX[tuple(sorted(p))] for p in pairs]


TRIANGLE_BITMASKS = []
for tri in TRIANGLES:
    mask = 0
    for bit in triangle_edge_bits(tri):
        mask |= (1 << bit)
    TRIANGLE_BITMASKS.append(mask)


def classical_marked_set():
    """Brute force over all 64 graphs on 4 labeled vertices (6 possible
    edges). Returns the sorted list of graph-encodings n in [0,64) whose
    edge set contains at least one complete triangle."""
    marked = []
    for n in range(64):
        has_triangle = False
        for mask in TRIANGLE_BITMASKS:
            if (n & mask) == mask:
                has_triangle = True
                break
        if has_triangle:
            marked.append(n)
    return marked


def build_oracle(n_qubits, marked):
    diag = np.ones(2 ** n_qubits, dtype=complex)
    for m in marked:
        diag[m] = -1.0
    return DiagonalGate(diag.tolist())


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
    marked = classical_marked_set()
    n_total = 64
    n_marked = len(marked)
    print(f"Classical brute force: {n_marked} of {n_total} labeled graphs "
          f"on 4 vertices contain at least one triangle.")
    print(f"Marked graph encodings (first 10 shown): {marked[:10]} ...")

    n_qubits = 6
    assert n_total == 2 ** n_qubits

    # Standard optimal-iteration formula, floored at 1 iteration.
    iterations = max(1, int(round((np.pi / 4) * np.sqrt(n_total / n_marked))))
    print(f"Grover iterations used: {iterations}")

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle, range(n_qubits))
        qc.append(diffuser, range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    shots = 20000
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Empirically verified against Statevector.probabilities() (which indexes
    # basis state i by qubit-0-is-least-significant-bit, i.e. no reversal of
    # the returned bitstring is needed to match that same integer index n):
    # interpreting the measured bitstring directly as a binary integer n
    # reproduces the statevector's per-outcome probabilities to within shot
    # noise, whereas reversing it does not.
    outcome_counts = {}
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        outcome_counts[n] = outcome_counts.get(n, 0) + c

    marked_set = set(marked)
    hits_on_marked = sum(c for n, c in outcome_counts.items() if n in marked_set)
    frac_marked = hits_on_marked / shots

    top_outcomes = sorted(outcome_counts.items(), key=lambda kv: -kv[1])[:n_marked]
    top_all_marked = all(n in marked_set for n, _ in top_outcomes)

    print(f"Fraction of shots landing on a classically-marked (triangle) "
          f"graph: {frac_marked:.4f}")
    print(f"Top {n_marked} most-measured outcomes are all classically "
          f"triangle-containing graphs: {top_all_marked}")

    # Theoretical maximum success probability for M=23 marked out of N=64
    # with the optimal integer iteration count is sin^2(3*theta) ~= 0.877
    # (theta = arcsin(sqrt(23/64))); require getting close to that bound.
    success = frac_marked > 0.85 and top_all_marked

    if success:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
