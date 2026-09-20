"""
Erdos problem #183 -- quantum-testable instance.

Source metadata (data/problems.yaml, manman4/erdosproblems clone):
  number: "183", tags: ["graph theory", "ramsey theory"],
  oeis: ["A003323"] (the formal Lean proof file is literally named
  MulticolorTriangleRamsey.lean), status: solved.

A003323 is the sequence of diagonal (multicolor, all-triangle) Ramsey
numbers R_k(3): the smallest n such that *every* k-coloring of the edges
of the complete graph K_n contains a monochromatic triangle. The classic
2-color term of this family is the ordinary Ramsey number R(3,3) = 6:
  - every 2-coloring of K_6's edges contains a monochromatic triangle,
  - but K_5 (and hence its subgraph K_4) admits at least one 2-coloring
    with NO monochromatic triangle.

Classical property tested here (finite, small, and computed from first
principles in this script, not copied from OEIS):

    Does K_4 (6 edges, 4 triangles) admit a 2-coloring of its edges with
    no monochromatic triangle?

K_4 has 6 edges -> 2^6 = 64 possible 2-colorings, a natural size for a
Grover search circuit (6 "edge" qubits). The oracle marks a coloring as
"good" iff none of the 4 triangles of K_4 is monochromatic (all three of
its edges the same color).

Vertices: 0,1,2,3. Edges (qubit index -> edge):
  q0=(0,1) q1=(0,2) q2=(0,3) q3=(1,2) q4=(1,3) q5=(2,3)
Triangles (as edge-qubit triples):
  {0,1,3} = (01,02,12) ; {0,2,4} = (01,03,13)
  {1,2,5} = (02,03,23) ; {3,4,5} = (12,13,23)

This script:
  1. Brute-forces all 64 colorings classically to get the exact count and
     list of "good" (triangle-free-coloring) solutions -- ground truth.
  2. Builds a real Grover search circuit (oracle + diffuser, iterated the
     optimal number of times for the known solution count) over the same
     6-qubit space, using qiskit_aer's AerSimulator (statevector, ideal).
  3. Runs the circuit, takes the most frequently measured 6-bit string,
     and checks classically that it is indeed a "good" (mono-triangle-free)
     coloring of K_4 -- i.e. the quantum search found a real solution.
  4. Prints PASS/FAIL based on agreement with the classical ground truth.
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute force all 2-colorings of K_4's 6 edges.
# ---------------------------------------------------------------------------

TRIANGLES = [(0, 1, 3), (0, 2, 4), (1, 2, 5), (3, 4, 5)]


def is_good_coloring(bits):
    """bits: tuple of 6 ints (0/1), one per edge. True iff no monochromatic
    triangle among TRIANGLES."""
    for a, b, c in TRIANGLES:
        if bits[a] == bits[b] == bits[c]:
            return False
    return True


all_colorings = list(product([0, 1], repeat=6))
good_colorings = [c for c in all_colorings if is_good_coloring(c)]

CLASSICAL_GOOD_SET = {c for c in good_colorings}
NUM_GOOD = len(good_colorings)

print(f"Classical brute force: {NUM_GOOD} / {len(all_colorings)} colorings "
      f"of K_4's edges have no monochromatic triangle.")
assert NUM_GOOD > 0, "K_4 must admit a triangle-free 2-coloring (R(3,3)=6 > 4)."

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle + diffuser over 6 "edge" qubits.
# ---------------------------------------------------------------------------

N = 6  # edge qubits: q0..q5
NUM_TRIANGLES = len(TRIANGLES)

# Register layout: 6 edge qubits, NUM_TRIANGLES ancilla "mono" flags,
# 1 output/phase-kickback qubit.
edge_qubits = list(range(N))
mono_anc = list(range(N, N + NUM_TRIANGLES))
out_qubit = N + NUM_TRIANGLES
total_qubits = N + NUM_TRIANGLES + 1


def mono_flag_circuit(qc, a, b, c, anc, m1, m2):
    """Correct, explicit construction of anc = 1 iff qubits a,b,c equal.
    Uses two helper 'mismatch' ancillas m1, m2 (each returned to 0)."""
    # m1 = a XOR b ; m2 = b XOR c
    qc.cx(a, m1)
    qc.cx(b, m1)
    qc.cx(b, m2)
    qc.cx(c, m2)
    # anc = NOT(m1) AND NOT(m2)  == equal(a,b) AND equal(b,c)
    qc.x(m1)
    qc.x(m2)
    qc.ccx(m1, m2, anc)
    # uncompute m1, m2 back to |0>
    qc.x(m1)
    qc.x(m2)
    qc.cx(c, m2)
    qc.cx(b, m2)
    qc.cx(b, m1)
    qc.cx(a, m1)


# Register sizes include 2 reusable helper qubits for the equality checks
# (shared/reused across the 4 triangle checks).
HELP = [total_qubits, total_qubits + 1]
TOTAL = total_qubits + 2


def build_full_oracle():
    qc = QuantumCircuit(TOTAL, name="oracle")
    m1, m2 = HELP
    for (a, b, c), anc in zip(TRIANGLES, mono_anc):
        mono_flag_circuit(qc, a, b, c, anc, m1, m2)
    # Mark "good" states: ALL mono ancillas are 0 (no monochromatic
    # triangle). Flip the phase of the output qubit (prepared in |-> by
    # the caller) controlled on all mono_anc == 0, i.e. flip X on each
    # mono_anc first, then a multi-controlled X onto out_qubit, then
    # undo the X's.
    for anc in mono_anc:
        qc.x(anc)
    qc.append(MCXGate(NUM_TRIANGLES), mono_anc + [out_qubit])
    for anc in mono_anc:
        qc.x(anc)
    # uncompute the mono ancillas
    for (a, b, c), anc in zip(TRIANGLES, mono_anc):
        mono_flag_circuit(qc, a, b, c, anc, m1, m2)
    return qc


# ---------------------------------------------------------------------------
# Assemble the full Grover circuit (oracle uses phase kickback on out_qubit
# prepared in |->; diffuser inverts about the mean on edge qubits using a
# standard H / X / MCX / X / H multi-controlled-Z construction).
# ---------------------------------------------------------------------------

def diffuser_mcz(qc):
    """Inversion about the mean over `edge_qubits`, implemented as an
    (N-1)-controlled Z via H + MCX + H on a dedicated target, reusing
    out_qubit is unsafe (it's entangled with the oracle's phase trick),
    so we implement the standard all-X / H / MCX(controls=N-1,target=last)
    / H / all-X construction directly on the edge register."""
    qc.h(edge_qubits)
    qc.x(edge_qubits)
    qc.h(edge_qubits[-1])
    qc.append(MCXGate(N - 1), edge_qubits[:-1] + [edge_qubits[-1]])
    qc.h(edge_qubits[-1])
    qc.x(edge_qubits)
    qc.h(edge_qubits)


oracle = build_full_oracle()

# Optimal number of Grover iterations for NUM_GOOD solutions out of 2**N.
theta = np.arcsin(np.sqrt(NUM_GOOD / 2 ** N))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
print(f"Grover iterations chosen: {iterations} (for {NUM_GOOD} solutions "
      f"out of {2 ** N})")

qc = QuantumCircuit(TOTAL, N)
qc.h(edge_qubits)
qc.x(out_qubit)
qc.h(out_qubit)

for _ in range(iterations):
    qc.compose(oracle, qubits=range(TOTAL), inplace=True)
    diffuser_mcz(qc)

qc.measure(edge_qubits, list(range(N)))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order: classical bit string is c[N-1]...c[0]; our measure
# mapped edge_qubits[i] -> classical bit i, so reverse for readability.
best_bitstring = max(counts, key=counts.get)
# best_bitstring[::-1][i] corresponds to edge_qubits[i]
measured_edges = tuple(int(b) for b in best_bitstring[::-1])

top_prob = counts[best_bitstring] / shots
print(f"Most frequent measured coloring: {measured_edges} "
      f"(probability {top_prob:.3f} over {shots} shots)")

# ---------------------------------------------------------------------------
# 4. Verify against the classical ground truth.
# ---------------------------------------------------------------------------

quantum_found_good = measured_edges in CLASSICAL_GOOD_SET

# Also check total probability mass landed on good colorings, as a
# sanity check that Grover actually amplified the correct subspace.
good_bitstrings = set()
for c in good_colorings:
    # convert to the measured bitstring convention (reversed)
    good_bitstrings.add("".join(str(b) for b in c[::-1]))
mass_on_good = sum(v for k, v in counts.items() if k in good_bitstrings) / shots

print(f"Total measured probability mass on good (triangle-free) colorings: "
      f"{mass_on_good:.3f}")

verified = quantum_found_good and mass_on_good > 0.5

if verified:
    print("PASS")
else:
    print("FAIL")
