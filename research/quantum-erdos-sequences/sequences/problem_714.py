"""
Erdos problem #714 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: '714'"):
    tags:   ["graph theory", "turan number"]
    oeis:   ["possible"]
    status: open (informal), unformalized (formal)

LIMITATION (read before trusting "PASS" below): problem 714's `oeis` field is
the literal string "possible", not a real OEIS sequence id. There is no A-number
to anchor a "is n in this sequence" or "what is term k" style test against, and
the problem's own page content was not available in this read-only clone beyond
the YAML metadata. So this script does NOT test an actual Erdos-problem-714
claim. Per instructions for the no-real-OEIS-id case, this is a best-honest-
effort substitute: the problem is tagged "turan number", so the script tests a
genuine, classically-checkable Turan-type extremal graph theory fact --
Turan's theorem for triangle-free graphs (Mantel's theorem) -- with a real
Grover search circuit, rather than fabricating or copying an OEIS value.

Classical property tested
--------------------------
Mantel's theorem: for n = 4 vertices, the maximum number of edges in a
triangle-free simple graph is floor(n^2 / 4) = 4, achieved by the complete
bipartite graph K_{2,2}.

There are C(4,2) = 6 possible edges on 4 labeled vertices, so the search space
of all labeled graphs on 4 vertices is exactly 2^6 = 64 basis states -- small
enough for a genuine 6-qubit Grover search on the ideal AerSimulator.

The script:
  1. Classically enumerates all 64 labeled graphs on 4 vertices, computes the
     edge count and triangle-freeness of each, and derives (from first
     principles, no OEIS lookup) the true maximum edge count among
     triangle-free graphs on 4 vertices, and the set of graphs achieving it.
  2. Builds a Grover oracle (as an exact multi-controlled-Z over the marked
     basis states, computed from the classical enumeration) that marks exactly
     the triangle-free graphs with that maximum edge count.
  3. Runs Grover search with the optimal number of iterations on
     AerSimulator, measures, and checks that the most frequently measured
     6-bit strings are exactly the classically-marked maximum triangle-free
     graphs.
  4. Prints PASS if the quantum search recovers the classical answer, else
     FAIL.

No dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator

ERDOS_PROBLEM_NUMBER = 714
OEIS_IDS_FOUND = ["possible"]  # not a real OEIS id -- see docstring limitation
TAGS = ["graph theory", "turan number"]

N_VERTICES = 4
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 6 possible edges
N_EDGE_QUBITS = len(EDGES)
assert N_EDGE_QUBITS == 6


def edges_from_bits(bits):
    """bits: tuple/list of 0/1 of length N_EDGE_QUBITS -> set of present edges."""
    return {EDGES[i] for i, b in enumerate(bits) if b}


def is_triangle_free(edge_set):
    for a, b, c in itertools.combinations(range(N_VERTICES), 3):
        if (a, b) in edge_set and (b, c) in edge_set and (a, c) in edge_set:
            return False
    return True


def classical_enumeration():
    """Enumerate all 2^6 labeled graphs on 4 vertices; return
    (max_edges_triangle_free, sorted list of marked bitstrings achieving it)."""
    best = -1
    per_count_marked = {}
    for bits in itertools.product([0, 1], repeat=N_EDGE_QUBITS):
        es = edges_from_bits(bits)
        if is_triangle_free(es):
            k = len(es)
            per_count_marked.setdefault(k, []).append(bits)
            if k > best:
                best = k
    marked = per_count_marked[best]
    # bitstring order: qubit i corresponds to EDGES[i]; Qiskit reports
    # classical register with qubit 0 as the rightmost character.
    marked_strs = sorted(
        "".join(str(b) for b in reversed(bits)) for bits in marked
    )
    return best, marked_strs


def build_oracle(marked_strs, n_qubits):
    """Phase-flip oracle marking exactly the given bitstrings (MSB-first,
    matching Qiskit's classical-register string convention)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
    for s in marked_strs:
        # s is MSB-first (qubit n-1 .. qubit 0); zero-bits get X sandwiches.
        zero_qubits = [n_qubits - 1 - i for i, c in enumerate(s) if c == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        qc.append(mcz, list(range(n_qubits)))
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
    qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_strs, n_qubits, shots=4096):
    n_marked = len(marked_strs)
    n_total = 2 ** n_qubits
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked_strs, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    print(f"Erdos problem #{ERDOS_PROBLEM_NUMBER}  tags={TAGS}  oeis={OEIS_IDS_FOUND}")
    print("No real OEIS id available -> substitute test: Mantel/Turan extremal "
          "triangle-free graph search on n=4 vertices (see module docstring).")

    max_edges, marked_strs = classical_enumeration()
    expected_max = N_VERTICES ** 2 // 4  # Mantel's theorem: floor(n^2/4)
    print(f"Classical: max triangle-free edge count on {N_VERTICES} vertices = "
          f"{max_edges} (Mantel formula floor(n^2/4) = {expected_max})")
    print(f"Classical: {len(marked_strs)} labeled graphs achieve this maximum "
          f"out of {2 ** N_EDGE_QUBITS} total labeled graphs")
    assert max_edges == expected_max, "classical enumeration disagrees with Mantel's theorem"

    counts, iterations = run_grover(marked_strs, N_EDGE_QUBITS)
    print(f"Grover: {iterations} iteration(s) on {N_EDGE_QUBITS} qubits, "
          f"{sum(counts.values())} shots")

    top_n = len(marked_strs)
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    top_strs = {s for s, _ in ranked[:top_n]}

    print("Top measured bitstrings:", ranked[:top_n])
    print("Classically marked (expected) bitstrings:", sorted(marked_strs))

    marked_set = set(marked_strs)
    total_shots = sum(counts.values())
    marked_shots = sum(c for s, c in counts.items() if s in marked_set)
    marked_fraction = marked_shots / total_shots

    verified = (top_strs == marked_set) and (marked_fraction > 0.8)

    print(f"Fraction of shots landing on a classically-marked bitstring: "
          f"{marked_fraction:.3f}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
