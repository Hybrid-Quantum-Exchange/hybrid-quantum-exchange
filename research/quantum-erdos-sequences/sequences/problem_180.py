"""
Erdos problem #180 (see /home/user/manman4/erdosproblems/data/problems.yaml,
"number: '180'"). Tags: ["graph theory", "turan number"]. oeis: ["N/A"] --
this problem has no associated OEIS sequence (the field is literally the
string "N/A" in the source data), so there is no OEIS id to derive a
property from. Per the task instructions, this is written as the best
honest attempt at a genuine, finite, computable property drawn from the
problem's own subject matter (Turan numbers / triangle-free graphs) rather
than from an OEIS entry, and the limitation (no OEIS id available) is
reported honestly rather than faking an OEIS-derived pass.

Classical property being tested
--------------------------------
Turan's theorem for triangles (K3): the maximum number of edges in a
triangle-free graph on n=4 vertices is ex(4, K3) = floor(4^2/4) = 4,
achieved uniquely (up to labeling) by the complete bipartite graph K_{2,2}
(equivalently: any 4-vertex, 4-edge graph obtained by deleting a perfect
matching from K4).

We enumerate all 2^6 = 64 subsets of the 6 edges of K4 (vertices 0,1,2,3;
edges e0=(0,1), e1=(0,2), e2=(0,3), e3=(1,2), e4=(1,3), e5=(2,3)) and, from
first principles, classically compute the set S of edge-subsets that (a)
contain exactly 4 edges and (b) contain none of the 4 possible triangles.
Turan's theorem predicts |S| = 3 (the three graphs K4 minus each of the
three perfect matchings of K4). We verify this classically in code below
(not copied from OEIS or the literature) and then have a real quantum
circuit (Grover search) search the same 64-element space for exactly these
solutions.

Quantum circuit
----------------
6 qubits represent the presence/absence of each of the 6 edges. A
reversible oracle, built from Toffoli (CCX) gates and Qiskit's
WeightedAdder, computes:
  - the Hamming weight (edge count) of the 6-qubit edge register via
    WeightedAdder, and marks whether it equals 4;
  - for each of the 4 possible triangles, the AND of its 3 edge qubits
    (i.e. whether that triangle is fully present), via two Toffolis per
    triangle with one shared, uncomputed temp ancilla;
  - a phase flip (via a target ancilla prepared in the |-> state) exactly
    when edge-count == 4 AND all four triangle-present flags are 0, i.e.
    exactly on triangle-free 4-edge graphs.

All working ancillas are uncomputed (gates applied, then inverse-applied)
so only the phase kickback survives, giving a valid Grover oracle. Grover
diffusion is applied on the 6-qubit edge register for the optimal number
of iterations for N=64, M=3 solutions, then the edge register is measured
on the ideal AerSimulator. PASS means the quantum circuit's most likely
measured outcomes are exactly the classically-computed solution set S.
"""

import itertools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import WeightedAdder
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # e0..e5
TRIANGLES = [
    (0, 1, 3),  # vertices {0,1,2} -> edges e0,e1,e3
    (0, 2, 4),  # vertices {0,1,3} -> edges e0,e2,e4
    (1, 2, 5),  # vertices {0,2,3} -> edges e1,e2,e5
    (3, 4, 5),  # vertices {1,2,3} -> edges e3,e4,e5
]


def is_triangle_free_with_4_edges(bits):
    """bits: tuple of 6 ints (0/1), bits[i] = presence of EDGES[i]."""
    if sum(bits) != 4:
        return False
    for (a, b, c) in TRIANGLES:
        if bits[a] and bits[b] and bits[c]:
            return False
    return True


classical_solutions = [
    bits for bits in itertools.product([0, 1], repeat=6)
    if is_triangle_free_with_4_edges(bits)
]

# Turan's theorem: ex(4, K3) = floor(4^2/4) = 4, achieved by exactly the 3
# graphs K4-minus-a-perfect-matching.
EXPECTED_TURAN_NUMBER = 4
EXPECTED_NUM_EXTREMAL_GRAPHS = 3

assert all(sum(b) == EXPECTED_TURAN_NUMBER for b in classical_solutions)
assert len(classical_solutions) == EXPECTED_NUM_EXTREMAL_GRAPHS, (
    f"classical enumeration found {len(classical_solutions)} extremal "
    f"graphs, expected {EXPECTED_NUM_EXTREMAL_GRAPHS}"
)

# Qiskit bit ordering: qubit 0 is the least-significant bit, and printed
# bitstrings are big-endian (qubit n-1 first). Build the little-endian bit
# tuples so they compare directly against measured bitstrings reversed.
classical_solution_strings = {
    "".join(str(b) for b in reversed(bits)) for bits in classical_solutions
}

print("Classical extremal (Turan) 4-edge triangle-free edge-subsets of K4:")
for bits in classical_solutions:
    present = [EDGES[i] for i, v in enumerate(bits) if v]
    print(f"  edges present: {present}")
print(f"Turan number ex(4,K3) = {EXPECTED_TURAN_NUMBER}, "
      f"{EXPECTED_NUM_EXTREMAL_GRAPHS} extremal graphs "
      f"(bitstrings, edge-qubit order q5..q0): {sorted(classical_solution_strings)}")

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 6-edge space.
# ---------------------------------------------------------------------------

edge = QuantumRegister(6, "edge")
adder = WeightedAdder(6, [1] * 6)  # sum(3) + carry(2) + control(1) ancillas
sum_reg = QuantumRegister(3, "sum")
carry_reg = QuantumRegister(2, "carry")
ctrl_reg = QuantumRegister(1, "actrl")
temp = QuantumRegister(1, "temp")
tri = QuantumRegister(4, "tri")
phase = QuantumRegister(1, "phase")
creg = ClassicalRegister(6, "meas")

qc = QuantumCircuit(edge, sum_reg, carry_reg, ctrl_reg, temp, tri, phase, creg)

# Uniform superposition over the 6 edge qubits.
qc.h(edge)

# Phase ancilla in the |-> state for standard phase-kickback marking.
qc.x(phase)
qc.h(phase)


def apply_oracle(circuit):
    # (a) Hamming weight of the edge register into sum_reg (WeightedAdder
    #     acts on [state..., sum..., carry..., control...] in that order).
    circuit.append(adder, edge[:] + sum_reg[:] + carry_reg[:] + ctrl_reg[:])

    # (b) AND of each triangle's 3 edges into tri[i], via a shared temp
    #     ancilla, uncomputed after each triangle.
    for i, (a, b, c) in enumerate(TRIANGLES):
        circuit.ccx(edge[a], edge[b], temp[0])
        circuit.ccx(temp[0], edge[c], tri[i])
        circuit.ccx(edge[a], edge[b], temp[0])  # uncompute temp

    # (c) Mark (phase-flip) when sum_reg == 4 (binary 100, i.e. bit2=1,
    #     bit1=0, bit0=0) AND all four tri[i] == 0.
    circuit.x(sum_reg[0])
    circuit.x(sum_reg[1])
    circuit.x(tri[0])
    circuit.x(tri[1])
    circuit.x(tri[2])
    circuit.x(tri[3])
    controls = [sum_reg[2], sum_reg[1], sum_reg[0],
                tri[0], tri[1], tri[2], tri[3]]
    circuit.mcx(controls, phase[0])
    circuit.x(sum_reg[0])
    circuit.x(sum_reg[1])
    circuit.x(tri[0])
    circuit.x(tri[1])
    circuit.x(tri[2])
    circuit.x(tri[3])

    # Uncompute (b) and (a) so only the phase kickback survives.
    for i, (a, b, c) in reversed(list(enumerate(TRIANGLES))):
        circuit.ccx(edge[a], edge[b], temp[0])
        circuit.ccx(temp[0], edge[c], tri[i])
        circuit.ccx(edge[a], edge[b], temp[0])
    circuit.append(adder.inverse(), edge[:] + sum_reg[:] + carry_reg[:] + ctrl_reg[:])


def apply_diffusion(circuit):
    circuit.h(edge)
    circuit.x(edge)
    circuit.h(edge[5])
    circuit.mcx(edge[0:5], edge[5])
    circuit.h(edge[5])
    circuit.x(edge)
    circuit.h(edge)


N = 64
M = EXPECTED_NUM_EXTREMAL_GRAPHS
import math
iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations: {iterations}")

for _ in range(iterations):
    apply_oracle(qc)
    apply_diffusion(qc)

qc.measure(edge, creg)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_m = {bs for bs, _ in sorted_counts[:M]}

print("\nTop measured bitstrings (edge-qubit order q5..q0) and counts:")
for bs, c in sorted_counts[:8]:
    tag = "SOLUTION" if bs in classical_solution_strings else ""
    print(f"  {bs}: {c} {tag}")

mass_on_solutions = sum(c for bs, c in counts.items() if bs in classical_solution_strings)
solution_fraction = mass_on_solutions / shots

verified = (top_m == classical_solution_strings) and (solution_fraction > 0.5)

print(f"\nFraction of shots landing on a classical Turan-extremal solution: "
      f"{solution_fraction:.3f}")
print(f"Top-{M} measured bitstrings match classical solution set exactly: "
      f"{top_m == classical_solution_strings}")

if verified:
    print("PASS")
else:
    print("FAIL")
