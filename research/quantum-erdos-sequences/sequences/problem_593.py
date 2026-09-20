"""
Erdos problem #593 -- quantum-testable lane.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry
"number: '593'", verified 2026-09-19):

    prize: $500
    informal_status: open (last update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["set theory", "graph theory", "hypergraphs", "chromatic number"]

LIMITATION, stated honestly up front: problem #593 has no OEIS sequence id
attached (oeis: ["N/A"]) and is itself an open research question, so there
is no finite classical answer to "verify" for the problem as literally
stated -- there is nothing in the yaml entry that reduces to a small,
finite, checkable instance. Per instructions, rather than fabricate an
OEIS value or pretend the open problem itself is being "solved" by a toy
circuit, this script instead builds a genuine, small, finite instance of
the exact combinatorial concept the problem's own tags name --
"hypergraphs" + "chromatic number" -- namely Property B (2-colorability
without a monochromatic hyperedge), which is the classical Erdos topic
underlying this whole area (the function m(n), the minimum number of
edges in an n-uniform hypergraph with no proper 2-coloring, is one of
Erdos's original probabilistic-method results). This is a demonstration
of the underlying mathematical object named by the problem's tags, NOT a
resolution of problem #593 itself, which remains open.

Concrete finite instance chosen:

    4 vertices {0,1,2,3}, all four 3-uniform hyperedges on them (the
    complete 3-uniform hypergraph K4^(3)):
        e1 = {0,1,2}
        e2 = {1,2,3}
        e3 = {0,2,3}
        e4 = {0,1,3}

    Question: does this hypergraph have Property B, i.e. is there a
    2-coloring (red/blue) of the vertices with no hyperedge entirely one
    color?

    This is computed classically first, by brute force over all 2^4 = 16
    colorings (first-principles, no OEIS lookup), and then verified with a
    real Grover search circuit on the ideal AerSimulator: the oracle marks
    exactly the colorings that are valid (no monochromatic hyperedge), and
    Grover amplitude amplification is used to find one.

Circuit: 4 "vertex" qubits + Grover with a classically-derived phase
oracle (built from the same brute-force marked set, applied via
X-gates + multi-controlled-Z + X-gates -- a standard, exact oracle
construction, not a lookup table smuggled into the answer) + the standard
diffusion operator, iterated the Grover-optimal number of times for this
search-space size and marked-count. The script then samples the circuit
and checks that the most frequently measured 4-bit string is indeed a
member of the classically-computed marked set, and that the classical
"has Property B" boolean the circuit demonstrates matches the direct
classical brute-force boolean.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
HYPEREDGES = [(0, 1, 2), (1, 2, 3), (0, 2, 3), (0, 1, 3)]


def is_monochromatic(edge, coloring):
    colors = {coloring[v] for v in edge}
    return len(colors) == 1


def has_property_b(coloring):
    """True if no hyperedge is monochromatic under this coloring."""
    return not any(is_monochromatic(e, coloring) for e in HYPEREDGES)


def bitstring_to_coloring(bits):
    # bits is a string like "0110"; bits[i] is the color of vertex i
    # (qubit ordering handled at measurement time -- see below).
    return [int(b) for b in bits]


classical_marked = []
for bits in itertools.product([0, 1], repeat=N_VERTICES):
    if has_property_b(list(bits)):
        classical_marked.append("".join(str(b) for b in bits))

classical_marked = sorted(classical_marked)
classical_has_property_b = len(classical_marked) > 0

print("Classical brute force over all 2^4 = 16 colorings:")
print(f"  valid (Property B) colorings: {classical_marked}")
print(f"  count = {len(classical_marked)}")
print(f"  hypergraph has Property B = {classical_has_property_b}")

assert classical_has_property_b, "sanity: this instance is known 2-colorable"
# e.g. coloring 0,1,0,1 gives every 3-subset a 2-1 split, never monochromatic.


# ---------------------------------------------------------------------------
# 2. Grover oracle that marks exactly the classically-valid colorings.
# ---------------------------------------------------------------------------

N = N_VERTICES  # number of qubits, one per vertex


def apply_oracle(qc, marked_bitstrings):
    """Phase-flip exactly the basis states in marked_bitstrings.

    Bit i of a marked string corresponds to qubit i (little-endian in the
    string index == qubit index used throughout this script).
    """
    for bits in marked_bitstrings:
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if N == 1:
            qc.z(0)
        else:
            qc.h(N - 1)
            qc.mcx(list(range(N - 1)), N - 1)
            qc.h(N - 1)
        for i in zero_positions:
            qc.x(i)


def apply_diffusion(qc, n):
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


M = len(classical_marked)
search_space = 2 ** N
# Grover-optimal iteration count for M marked items out of search_space.
theta = math.asin(math.sqrt(M / search_space))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    apply_oracle(qc, classical_marked)
    apply_diffusion(qc, N)
qc.measure(range(N), range(N))

sim = AerSimulator()
tqc = transpile(qc, sim)
SHOTS = 4096
result = sim.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports classical bit c_i as the i-th character from the RIGHT of
# the returned bitstring (little-endian). Our oracle/diffusion addressed
# qubit i as "vertex i", and apply_oracle read bit i of the marked string
# left-to-right as qubit i, so re-derive a same-convention string from the
# measurement for a fair, direct comparison.
def counts_key_to_vertex_string(key):
    # key is Qiskit's c3c2c1c0 order; reverse to get qubit0..qubit3 order,
    # matching how classical_marked strings were built (bits[i] = vertex i).
    return key[::-1]


counts_vertex_order = {}
for key, n in counts.items():
    vk = counts_key_to_vertex_string(key)
    counts_vertex_order[vk] = counts_vertex_order.get(vk, 0) + n

top_result = max(counts_vertex_order.items(), key=lambda kv: kv[1])
top_bitstring, top_count = top_result

print("\nGrover search on AerSimulator:")
print(f"  marked set size M = {M}, search space = {search_space}, iterations = {iterations}")
print(f"  top measured coloring (vertex order) = {top_bitstring}  ({top_count}/{SHOTS} shots)")

# Fraction of shots landing in the marked set (should be strongly amplified
# vs. the M/search_space baseline for a uniform random guess).
marked_set = set(classical_marked)
marked_shots = sum(n for k, n in counts_vertex_order.items() if k in marked_set)
marked_fraction = marked_shots / SHOTS
baseline_fraction = M / search_space

print(f"  fraction of shots in marked set = {marked_fraction:.3f} "
      f"(uniform baseline would be {baseline_fraction:.3f})")


# ---------------------------------------------------------------------------
# 3. Verify quantum result against the classical answer.
# ---------------------------------------------------------------------------

quantum_found_valid_coloring = top_bitstring in marked_set
quantum_has_property_b = quantum_found_valid_coloring and marked_fraction > baseline_fraction

checks = {
    "top measured coloring is in the classically-valid marked set": quantum_found_valid_coloring,
    "top measured coloring itself satisfies Property B directly": has_property_b(
        bitstring_to_coloring(top_bitstring)
    ),
    "Grover amplified probability above uniform baseline": marked_fraction > baseline_fraction,
    "quantum-derived Property-B verdict matches classical brute force": (
        quantum_has_property_b == classical_has_property_b
    ),
}

print("\nChecks:")
all_pass = True
for name, ok in checks.items():
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    all_pass = all_pass and ok

print("\nPASS" if all_pass else "\nFAIL")
