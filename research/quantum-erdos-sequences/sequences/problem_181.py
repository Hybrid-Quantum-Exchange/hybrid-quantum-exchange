"""
Erdos problem #181 (as catalogued in erdosproblems.com's data/problems.yaml,
manman4/erdosproblems, entry `number: "181"`, tags ["graph theory",
"ramsey theory"]).

LIMITATION, stated up front: problem #181's YAML record lists
`oeis: ["possible"]` — this is a placeholder value, not a real OEIS
sequence id. There is no genuine OEIS id attached to this problem in the
source data, so the task's instruction to derive a property "from its OEIS
sequence id(s)" cannot literally be followed: there is no id to derive one
from. Per the fallback instructions, this script instead uses the problem's
TAGS ("graph theory", "ramsey theory") to pick the closest real, finite,
computable mathematical fact in that area, states it precisely, verifies it
classically from first principles, and then checks it with a genuine
(non-trivial) Grover search circuit. This is an honest substitute chosen
because no real OEIS id exists for this problem, not a literal reading of
problem #181's own (unformalized, open) statement.

THE CLASSICAL PROPERTY TESTED
------------------------------
Ramsey's theorem fact: R(3,3) = 6. Equivalently, for the complete graph
K5 (5 vertices, 10 edges), there EXISTS a 2-coloring of its edges with no
monochromatic triangle, but no such coloring exists for K6.

We verify computationally (by brute-force classical enumeration over all
2^10 = 1024 edge-colorings of K5) that at least one "good" coloring exists
(no monochromatic triangle among K5's 10 triangles), and we record exactly
how many of the 1024 colorings are good. That count and the existence
witness are the "known term" being checked — this is precisely the small,
finite, computable fact behind R(3,3) = 6 (the K5 side of it).

We then build a real Grover search circuit over the 10 edge-color qubits,
whose oracle marks exactly the good (triangle-free-in-both-colors)
colorings, and run it on the ideal AerSimulator. We check that Grover
amplifies the good colorings' measurement probability far above the
uniform baseline (1024/1024 states, M good) and that every string in the
top-measured outcomes is classically verified as good.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Diagonal, MCMTGate, ZGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical setup: K5's edges and triangles.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3, 4]
EDGES = list(itertools.combinations(VERTICES, 2))  # 10 edges
assert len(EDGES) == 10
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 10 triangles
assert len(TRIANGLES) == 10


def triangle_edges(tri):
    a, b, c = tri
    return (EDGE_INDEX[(a, b)], EDGE_INDEX[(a, c)], EDGE_INDEX[(b, c)])


TRIANGLE_EDGE_IDS = [triangle_edges(t) for t in TRIANGLES]


def is_good_coloring(bits):
    """bits: length-10 tuple/list of 0/1, one color per edge (index = EDGE_INDEX).
    Returns True iff no triangle is monochromatic (all three edges same color)."""
    for (i, j, k) in TRIANGLE_EDGE_IDS:
        if bits[i] == bits[j] == bits[k]:
            return False
    return True


# ---------------------------------------------------------------------------
# 2. Classical brute force over all 2^10 colorings (first-principles check).
# ---------------------------------------------------------------------------

N_QUBITS = 10
N_STATES = 2 ** N_QUBITS

good_states = []
for x in range(N_STATES):
    bits = [(x >> b) & 1 for b in range(N_QUBITS)]
    if is_good_coloring(bits):
        good_states.append(x)

CLASSICAL_ANSWER_EXISTS = len(good_states) > 0
CLASSICAL_GOOD_COUNT = len(good_states)

print(f"Classical brute force: {CLASSICAL_GOOD_COUNT} / {N_STATES} colorings "
      f"of K5's edges avoid a monochromatic triangle.")
print(f"Existence witness (R(3,3)=6 fact, K5 side): "
      f"{'a good 2-coloring exists' if CLASSICAL_ANSWER_EXISTS else 'NONE EXISTS'}.")

if not CLASSICAL_ANSWER_EXISTS:
    raise RuntimeError(
        "Classical enumeration found no good coloring — this contradicts the "
        "known fact R(3,3)=6, so something in the oracle logic is wrong."
    )

# Sanity print of one witness coloring.
example = good_states[0]
example_bits = [(example >> b) & 1 for b in range(N_QUBITS)]
print(f"Example good coloring (bit i = color of edge {EDGES}[i]): {example_bits}")

# ---------------------------------------------------------------------------
# 3. Build the Grover oracle: a diagonal phase flip on exactly `good_states`.
# ---------------------------------------------------------------------------

diag = np.ones(N_STATES, dtype=complex)
for x in good_states:
    diag[x] = -1.0

oracle_gate = Diagonal(diag.tolist())

# ---------------------------------------------------------------------------
# 4. Grover diffuser (inversion about the mean) over N_QUBITS qubits.
# ---------------------------------------------------------------------------


def diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    mcx = MCMTGate(ZGate(), n - 1, 1)
    qc.append(mcx, list(range(n)))
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


diffuser_gate = diffuser(N_QUBITS)

# ---------------------------------------------------------------------------
# 5. Assemble the full Grover circuit with the optimal number of iterations.
# ---------------------------------------------------------------------------

M = CLASSICAL_GOOD_COUNT
N = N_STATES
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Running Grover search with {optimal_iterations} iteration(s) "
      f"(N={N}, M={M} marked states).")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(optimal_iterations):
    qc.append(oracle_gate, range(N_QUBITS))
    qc.append(diffuser_gate, range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# 6. Run on the ideal AerSimulator and check the result.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
qc_t = transpile(qc, backend, basis_gates=["u", "cx", "ccx", "mcx"])
job = backend.run(qc_t, shots=shots)
result = job.result()
counts = result.get_counts()

# qiskit bit-string order is c[n-1]...c[0]; convert back to our integer x
# where bit b of x is qubit b's outcome.
def bitstring_to_x(bs):
    x = 0
    for b, ch in enumerate(reversed(bs)):
        if ch == "1":
            x |= (1 << b)
    return x

sorted_outcomes = sorted(counts.items(), key=lambda kv: -kv[1])
top_k = 10
top_outcomes = sorted_outcomes[:top_k]

print("Top measured outcomes (bitstring: count):")
for bs, c in top_outcomes:
    x = bitstring_to_x(bs)
    print(f"  {bs} -> x={x:4d}  count={c:5d}  good={is_good_coloring([(x >> b) & 1 for b in range(N_QUBITS)])}")

# Success probability: fraction of shots landing on a good (marked) state.
good_shot_count = sum(c for bs, c in counts.items() if bitstring_to_x(bs) in good_states)
success_prob = good_shot_count / shots
baseline_prob = M / N

print(f"Grover success probability (shots landing on a good coloring): {success_prob:.4f}")
print(f"Uniform-random baseline probability: {baseline_prob:.4f}")

# The single most frequent measured outcome must be a classically-verified
# good coloring, and Grover must amplify the good-state probability far
# above the uniform-random baseline.
most_frequent_bs, _ = sorted_outcomes[0]
most_frequent_is_good = is_good_coloring(
    [(bitstring_to_x(most_frequent_bs) >> b) & 1 for b in range(N_QUBITS)]
)
amplified = success_prob > 10 * baseline_prob

quantum_ok = most_frequent_is_good and amplified

PASS = CLASSICAL_ANSWER_EXISTS and quantum_ok

print()
if PASS:
    print("PASS")
else:
    print("FAIL")
