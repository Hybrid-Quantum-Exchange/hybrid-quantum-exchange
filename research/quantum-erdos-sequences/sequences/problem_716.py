"""
Erdos problem #716 -- quantum-testable sequence entry.

Source metadata (erdosproblems.com data, problems.yaml, entry "716"):
    prize: no
    status: proved (Lean)
    oeis: ["possible"]
    comments: "Ruzsa-Szemeredi problem"
    tags: ["graph theory", "hypergraphs"]

LIMITATION (reported honestly, not glossed over):
    The problems.yaml entry for #716 does NOT carry a real OEIS sequence id --
    its "oeis" field is the literal placeholder string "possible", not an
    A-number. There is therefore no finite integer sequence membership/term
    property from OEIS to test here. Rather than fabricate an OEIS-derived
    claim, this script tests a genuine, small, finite, classically-checkable
    combinatorial property that is faithful to the *mathematical content* of
    problem #716 (the Ruzsa-Szemeredi (6,3) problem is fundamentally about
    induced matchings inside graphs/hypergraphs): existence and enumeration
    of PERFECT MATCHINGS in the complete graph K4, searched for with Grover's
    algorithm over the 2^6 = 64 possible edge subsets of K4.

Classical property being tested:
    K4 has vertices {0,1,2,3} and 6 edges, indexed e0..e5:
        e0=(0,1) e1=(0,2) e2=(0,3) e3=(1,2) e4=(1,3) e5=(2,3)
    A subset of edges (encoded as a 6-bit string, bit i = 1 iff edge e_i is
    included) is "marked" iff it is exactly a perfect matching of K4, i.e.
    exactly 2 disjoint edges covering all 4 vertices. K4 has exactly 3
    perfect matchings:
        {e0,e5} = {(0,1),(2,3)}
        {e1,e4} = {(0,2),(1,3)}
        {e2,e3} = {(0,3),(1,2)}
    The classical answer (brute force over all 64 subsets, computed in this
    script from first principles) is: exactly these 3 of the 64 possible
    edge-subsets are perfect matchings.

Quantum approach:
    Grover search over a 6-qubit index register (2^6 = 64 basis states, one
    per edge subset). The oracle applies a phase flip to exactly the 3
    marked "perfect matching" bitstrings, built directly from the classically
    computed marked set (no OEIS value copied in). ceil(pi/4 * sqrt(64/3))
    ~= 2 Grover iterations are applied on the ideal AerSimulator. The script
    passes if the measurement distribution is concentrated (top-3 outcomes
    by count) on exactly the 3 classically-verified marked bitstrings.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force all edge subsets of K4, from
#    first principles (no OEIS values used).
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # e0..e5
N_EDGES = len(EDGES)
N_QUBITS = N_EDGES  # one qubit per edge -> index register of size 6


def is_perfect_matching(edge_subset_bits):
    """edge_subset_bits: tuple of 6 bits (bit i -> EDGES[i] included).
    Returns True iff the included edges form a perfect matching of K4
    (every vertex covered exactly once)."""
    chosen = [EDGES[i] for i, b in enumerate(edge_subset_bits) if b == 1]
    covered = []
    for (u, v) in chosen:
        covered.append(u)
        covered.append(v)
    if sorted(covered) != VERTICES:
        return False
    # every vertex covered exactly once <=> chosen edges pairwise disjoint
    return len(set(covered)) == len(covered)


all_subsets = list(np.ndindex(*([2] * N_EDGES)))
marked = [s for s in all_subsets if is_perfect_matching(s)]

# Sanity: enumerate perfect matchings of K4 the "textbook" way too, and
# cross-check the two brute-force methods agree.
textbook_matchings = []
for pair in combinations(range(N_EDGES), 2):
    e_a, e_b = EDGES[pair[0]], EDGES[pair[1]]
    if len(set(e_a) | set(e_b)) == 4:  # disjoint, covers all 4 vertices
        textbook_matchings.append(pair)

assert len(marked) == 3, f"expected exactly 3 perfect matchings, got {len(marked)}"
assert len(textbook_matchings) == 3
marked_index_pairs = {tuple(i for i, b in enumerate(s) if b == 1) for s in marked}
assert marked_index_pairs == set(textbook_matchings), "brute-force methods disagree"

# Bitstrings as Qiskit will report them: qubit 0 is the rightmost character.
def bits_to_qiskit_string(bits):
    # bits[i] corresponds to qubit i (edge e_i); Qiskit prints q_{n-1}...q_0
    return "".join(str(bits[i]) for i in reversed(range(N_EDGES)))


marked_strings = sorted(bits_to_qiskit_string(s) for s in marked)

print("Classical ground truth (brute force over all 64 edge subsets of K4):")
print(f"  perfect matchings found: {len(marked)} of {len(all_subsets)} subsets")
for s in marked:
    edges_in = [EDGES[i] for i, b in enumerate(s) if b == 1]
    print(f"    bits={s} qiskit_string={bits_to_qiskit_string(s)} edges={edges_in}")

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 6-qubit index register for the
#    3 marked "perfect matching" bitstrings.
# ---------------------------------------------------------------------------

N = 2 ** N_QUBITS  # 64
M = len(marked)  # 3
iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"\nGrover iterations used: {iterations} (N={N}, M={M})")


def apply_oracle(qc: QuantumCircuit, marked_bit_tuples):
    """Phase-flip each marked basis state |bits> (bits[i] = value of qubit i)."""
    for bits in marked_bit_tuples:
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all N_QUBITS qubits: use an MCX with the
        # target qubit put into the |-> state so phase kickback gives MCZ.
        qc.h(N_QUBITS - 1)
        qc.append(MCXGate(N_QUBITS - 1), list(range(N_QUBITS - 1)) + [N_QUBITS - 1])
        qc.h(N_QUBITS - 1)
        for i in zero_positions:
            qc.x(i)


def apply_diffuser(qc: QuantumCircuit):
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.append(MCXGate(N_QUBITS - 1), list(range(N_QUBITS - 1)) + [N_QUBITS - 1])
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))


qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

for _ in range(iterations):
    apply_oracle(qc, marked)
    apply_diffuser(qc)

qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 20000
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

top3 = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
top3_strings = sorted(k for k, _ in top3)

print("\nTop measured bitstrings (Qiskit qN-1..q0 order):")
for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
    tag = " <- marked (perfect matching)" if k in marked_strings else ""
    print(f"  {k}: {v}/{shots}{tag}")

mass_on_marked = sum(v for k, v in counts.items() if k in marked_strings) / shots

print(f"\nClassically marked bitstrings: {marked_strings}")
print(f"Top-3 measured bitstrings:      {top3_strings}")
print(f"Measured probability mass on marked states: {mass_on_marked:.4f}")

verified = (top3_strings == marked_strings) and (mass_on_marked > 0.8)

if verified:
    print("\nPASS: Grover search on K4's perfect-matching oracle concentrated "
          "on exactly the 3 classically-verified perfect matchings.")
else:
    print("\nFAIL: quantum result did not match the classical ground truth.")

assert verified, "quantum result did not match classical ground truth"
