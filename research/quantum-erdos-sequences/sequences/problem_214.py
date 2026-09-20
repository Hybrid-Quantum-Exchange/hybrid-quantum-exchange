"""
Erdos problem #214 (erdosproblems.com / manman4/erdosproblems dataset).

Metadata for problem 214 in data/problems.yaml:
    prize: no
    status: proved (Lean)
    oeis: ["N/A"]
    tags: ["geometry", "distances", "ramsey theory"]

Problem 214 carries NO OEIS sequence id in the dataset (oeis: ["N/A"]), so
this script cannot test membership in, or a defining property of, an actual
OEIS sequence for this problem -- that would require fabricating an id that
is not in the source data, which the task explicitly forbids. This is the
honest limitation: there is no sequence to attach a quantum test to.

Best-effort substitute, chosen from the problem's own tags
("ramsey theory", "geometry", "distances"):

    Classical property tested: the diagonal Ramsey number R(3,3) = 6.
    Equivalently: there EXISTS a 2-coloring of the edges of the complete
    graph K5 (5 vertices, 10 edges) that contains no monochromatic
    triangle. (For K6 no such coloring exists -- that boundary, 5 vs 6,
    is exactly the classical fact R(3,3)=6, a canonical Ramsey-theory
    result of the same flavor as problem 214's tags.)

    This is a genuine finite, computable search problem: search space of
    2^10 = 1024 edge-colorings of K5, decide for each whether it avoids a
    monochromatic triangle among the C(5,3) = 10 triangles.

Classical ground truth (computed here from first principles, no lookup):
    Brute force over all 1024 colorings finds the exact set of "good"
    colorings (no monochromatic triangle). This script prints how many
    there are and picks one as the target the quantum search must find.

Quantum approach:
    Grover's algorithm on 10 qubits (one per edge of K5). The oracle is
    built directly from the classically-enumerated set of good colorings
    (marking each of them with a multi-controlled Z after the appropriate
    X-conjugation), so the oracle encodes exactly the classical "no
    monochromatic triangle" predicate -- nothing is hard-coded from a
    known answer beyond what the brute-force search above derived.
    Grover amplifies the good colorings; running the circuit on the ideal
    AerSimulator and taking the most frequent measured bitstring should
    land on one of the classically verified good colorings.

PASS/FAIL: the script measures the circuit, takes the most-frequent
outcome, and checks classically (independently, via the same
mono-triangle check) that it truly is a coloring with no monochromatic
triangle. That is the sense in which this verifies against the classical
answer -- honestly, this is a demonstration of Grover search on a genuine
combinatorial (Ramsey-type) predicate related to problem 214's tags, not
a literal OEIS-sequence test, because problem 214 has no OEIS id.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical setup: K5, its edges/triangles, and the "no monochromatic
#    triangle" predicate, derived from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 5
edges = list(itertools.combinations(range(N_VERTICES), 2))  # 10 edges
triangles = list(itertools.combinations(range(N_VERTICES), 3))  # 10 triangles
edge_index = {e: i for i, e in enumerate(edges)}
n_qubits = len(edges)
assert n_qubits == 10


def has_monochromatic_triangle(coloring):
    """coloring: tuple of 0/1, length 10, one bit per edge (in `edges` order)."""
    for a, b, c in triangles:
        e1 = edge_index[tuple(sorted((a, b)))]
        e2 = edge_index[tuple(sorted((a, c)))]
        e3 = edge_index[tuple(sorted((b, c)))]
        if coloring[e1] == coloring[e2] == coloring[e3]:
            return True
    return False


# Brute-force classical search over all 2^10 colorings.
good_colorings = []
for bits in itertools.product([0, 1], repeat=n_qubits):
    if not has_monochromatic_triangle(bits):
        good_colorings.append(bits)

n_good = len(good_colorings)
print(f"Classical brute force: {n_good} of {2 ** n_qubits} edge-colorings of "
      f"K5 avoid a monochromatic triangle.")
assert n_good > 0, "R(3,3)=6 implies K5 (unlike K6) must admit such a coloring"

# The classical answer this script is testing against: it is possible to
# 2-color the edges of K5 with no monochromatic triangle (R(3,3) = 6, i.e.
# the "no" boundary is at n=5, not n=6). Grover search must find a witness.


# ---------------------------------------------------------------------------
# 2. Quantum: Grover search whose oracle marks exactly `good_colorings`,
#    built straight from the classical enumeration above.
# ---------------------------------------------------------------------------

def bitstring_to_qiskit_order(bits):
    """qiskit uses little-endian qubit ordering in bitstrings; keep a plain
    tuple->list conversion, we index qubits directly by edge position."""
    return list(bits)


def build_oracle(marked_states, n):
    qc = QuantumCircuit(n, name="oracle")
    for state in marked_states:
        # state[i] corresponds to qubit i (edge i). Flip 0-bits to 1 so a
        # multi-controlled Z fires exactly on this state, then flip back.
        zero_qubits = [i for i, b in enumerate(state) if b == 0]
        for q in zero_qubits:
            qc.x(q)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(good_colorings, n_qubits)
diffuser = build_diffuser(n_qubits)

# Optimal number of Grover iterations for |marked|=n_good out of N=2^n_qubits.
N = 2 ** n_qubits
theta = math.asin(math.sqrt(n_good / N))
iterations = max(1, round((math.pi / 4) / theta - 0.5))
print(f"Using {iterations} Grover iteration(s) for {n_good}/{N} marked states.")

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(n_qubits), range(n_qubits))

sim = AerSimulator()
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=4096).result()
counts = result.get_counts()

# Most frequent measured bitstring. Qiskit's classical-register bitstring is
# big-endian in cbit index (c[n-1] ... c[0]); build our own tuple back in
# `edges` order (qubit i -> edge i) by reversing the printed string.
best_bitstring = max(counts, key=counts.get)
best_prob = counts[best_bitstring] / sum(counts.values())
measured_bits = tuple(int(b) for b in reversed(best_bitstring))  # qubit i first

print(f"Most frequent measurement: {best_bitstring} "
      f"(edge-coloring {measured_bits}), probability {best_prob:.3f} "
      f"over {len(counts)} distinct outcomes.")

# Sanity: how much amplitude landed on marked states overall.
marked_set = set(good_colorings)
marked_prob = sum(
    c for bs, c in counts.items()
    if tuple(int(b) for b in reversed(bs)) in marked_set
) / sum(counts.values())
print(f"Total measured probability on classically-verified good colorings: "
      f"{marked_prob:.3f} (uniform baseline would be {n_good / N:.3f}).")


# ---------------------------------------------------------------------------
# 3. Verify: independently re-check the measured winner classically.
# ---------------------------------------------------------------------------

quantum_found_good = measured_bits in marked_set
verified = (not has_monochromatic_triangle(measured_bits)) == quantum_found_good

ran_ok = True
if quantum_found_good and verified:
    print("PASS: Grover search on the K5 edge-2-coloring oracle (Ramsey "
          "R(3,3)=6 witness search, related to problem 214's tags; problem "
          "214 itself has no OEIS id) returned a coloring independently "
          "verified classically to have no monochromatic triangle.")
else:
    print("FAIL: most frequent measured outcome was not a classically "
          "verified monochromatic-triangle-free coloring of K5.")
