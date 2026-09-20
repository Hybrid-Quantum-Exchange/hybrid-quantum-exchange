"""
Erdos problem #565 (erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: \"565\"", tags ["graph theory", "ramsey theory"]).

LIMITATION, stated up front: problem 565's YAML entry lists oeis: ["possible"],
which is not an actual OEIS sequence id (unlike most entries, which list real
ids such as "A000040"). There is therefore no genuine OEIS sequence to build a
membership/term-defining circuit against for this problem. Rather than
fabricate an OEIS-derived property, this script honors the problem's own
subject matter instead: its tags are "graph theory" and "ramsey theory", and
Ramsey-type "does every 2-coloring of this graph's edges contain a
monochromatic triangle" questions are the classical prototype of the field
(the Ramsey number R(3,3)=6 says every 2-coloring of K6's edges has a
monochromatic triangle, and this is tight: K5 admits a coloring with none).

Chosen finite, computable property (not copied from any OEIS value):
  For the complete graph K4 (4 vertices, 6 edges, 4 triangles), consider all
  2^6 = 64 ways to 2-color its edges. A coloring is "triangle-avoiding" if none
  of its 4 triangles is monochromatic (all three edges the same color).
  Because R(3,3) = 6 > 4, such colorings must exist for K4 (indeed for K5
  too); this script first computes the exact set of triangle-avoiding
  colorings of K4 classically, by brute force over all 64 colorings, then uses
  a real Grover search circuit (built from a genuine quantum oracle that
  evaluates the "no monochromatic triangle" predicate reversibly, entirely in
  superposition) to find one, and checks the circuit's most probable output
  against the classically-verified set.

Qubits: 6 "edge" qubits (search register, N = 2^6 = 64) + 4 ancilla qubits
(one per triangle, each flags "this triangle is monochromatic") + 1 oracle
output qubit (phase kickback), reused across the fixed number of Grover
iterations. All logic (triangle checks, the AND-of-NOTs into the oracle
qubit, and uncomputation) is done with reversible gates -- no classical
shortcuts inside the circuit.
"""

import itertools
import math

from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges of K4
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles of K4
assert len(TRIANGLES) == 4


def triangle_edge_indices(tri):
    a, b, c = tri
    pairs = [(a, b), (a, c), (b, c)]
    return [EDGE_INDEX[tuple(sorted(p))] for p in pairs]


TRI_EDGE_IDX = [triangle_edge_indices(t) for t in TRIANGLES]


def is_triangle_avoiding(bits):
    """bits: tuple of 6 ints (0/1), one per edge in EDGES order."""
    for (i, j, k) in TRI_EDGE_IDX:
        if bits[i] == bits[j] == bits[k]:
            return False
    return True


classical_solutions = []
for combo in itertools.product([0, 1], repeat=6):
    if is_triangle_avoiding(combo):
        classical_solutions.append(combo)

NUM_SOLUTIONS = len(classical_solutions)
N = 64
assert NUM_SOLUTIONS > 0, "R(3,3)=6 > 4 guarantees triangle-avoiding colorings of K4 exist"

solution_ints = sorted(int("".join(map(str, s[::-1])), 2) for s in classical_solutions)
# (bit order matches Qiskit's little-endian qubit-to-bitstring convention,
#  where qubit 0 is the least significant bit)

print(f"Classical brute force over all {N} edge-colorings of K4:")
print(f"  triangle-avoiding colorings found: {NUM_SOLUTIONS}")
print(f"  as integers (qubit-0-is-LSB encoding): {solution_ints}")

# ---------------------------------------------------------------------------
# 2. Quantum oracle + Grover search circuit.
# ---------------------------------------------------------------------------

edge_reg = QuantumRegister(6, "edge")
tri_reg = QuantumRegister(4, "tri")
out_reg = QuantumRegister(1, "out")

qc = QuantumCircuit(edge_reg, tri_reg, out_reg, name="grover_k4_ramsey")

# Prepare oracle output qubit in |-> for phase kickback.
qc.x(out_reg[0])
qc.h(out_reg[0])

# Uniform superposition over all 64 edge colorings.
qc.h(edge_reg)

optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / NUM_SOLUTIONS) - 0.5))


def apply_triangle_flags(circuit):
    """For each triangle, flip its ancilla iff the 3 edges are monochromatic.

    mono(e0,e1,e2) = (e0==e1==e2) = NOT( (e0 XOR e1) OR (e0 XOR e2) ).
    We build this reversibly with CNOTs into two scratch positions reused
    across triangles via the ancilla itself: ancilla is set to 1 exactly when
    the triangle is monochromatic, using an X-CCX (Toffoli on XORs) pattern.
    """
    for t_idx, (i, j, k) in enumerate(TRI_EDGE_IDX):
        e0, e1, e2 = edge_reg[i], edge_reg[j], edge_reg[k]
        anc = tri_reg[t_idx]
        # anc starts at |0>. Compute e0==e1 AND e0==e2 into anc.
        # Use two CNOTs to test equality: flip e1,e2 copies via CNOT with e0,
        # store equality test in temporary use of anc via a CCX trick:
        # Standard construction: mono = NOT(e0 XOR e1) AND NOT(e0 XOR e2)
        qc_local = circuit
        qc_local.cx(e0, e1)  # e1 <- e0 XOR e1  (now e1==0 iff original e0==e1)
        qc_local.cx(e0, e2)  # e2 <- e0 XOR e2  (now e2==0 iff original e0==e2)
        qc_local.x(e1)
        qc_local.x(e2)
        qc_local.ccx(e1, e2, anc)  # anc = 1 iff both equalities held
        qc_local.x(e1)
        qc_local.x(e2)
        qc_local.cx(e0, e2)  # uncompute e2
        qc_local.cx(e0, e1)  # uncompute e1


def uncompute_triangle_flags(circuit):
    # The flag-setting above is its own inverse (all gates are self-inverse
    # CNOT/CCX/X in a palindromic arrangement), so re-applying it once more
    # after the flags have been consumed restores the ancillas to |0>.
    apply_triangle_flags(circuit)


def oracle(circuit):
    apply_triangle_flags(circuit)
    # Mark iff ALL 4 triangle ancillas are 0 (no monochromatic triangle):
    # flip each ancilla, multi-controlled-X into out qubit, flip back.
    for a in tri_reg:
        circuit.x(a)
    circuit.mcx(list(tri_reg), out_reg[0])
    for a in tri_reg:
        circuit.x(a)
    uncompute_triangle_flags(circuit)


def diffuser(circuit):
    circuit.h(edge_reg)
    circuit.x(edge_reg)
    circuit.h(edge_reg[-1])
    circuit.mcx(list(edge_reg[:-1]), edge_reg[-1])
    circuit.h(edge_reg[-1])
    circuit.x(edge_reg)
    circuit.h(edge_reg)


for _ in range(optimal_iterations):
    oracle(qc)
    diffuser(qc)

# Undo the |-> prep on the output qubit before measuring (not strictly
# necessary since we don't measure it, but keeps the circuit tidy/reversible
# in spirit).
qc.h(out_reg[0])
qc.x(out_reg[0])

creg_name = "meas"
qc.add_register(QuantumRegister(0))  # no-op, keeps structure explicit
from qiskit import ClassicalRegister
creg = ClassicalRegister(6, creg_name)
qc.add_register(creg)
qc.measure(edge_reg, creg)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
job = backend.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

# Most frequent measured bitstring -> integer (qubit 0 = LSB, matches how we
# built solution_ints above from Qiskit's little-endian convention).
best_bitstring = max(counts, key=counts.get)
best_int = int(best_bitstring, 2)

total_solution_shots = sum(c for bstr, c in counts.items() if int(bstr, 2) in solution_ints)
solution_fraction = total_solution_shots / shots

print(f"Grover iterations used: {optimal_iterations}")
print(f"Most frequent measured coloring (int): {best_int}")
print(f"Fraction of shots landing on a classically-valid solution: {solution_fraction:.3f}")

verified = (best_int in solution_ints) and (solution_fraction > 0.5)

if verified:
    print("PASS: Grover search's top result is a classically-verified "
          "triangle-avoiding 2-coloring of K4, and most shots concentrate on "
          "valid solutions.")
else:
    print("FAIL: quantum result did not match the classically-verified "
          "solution set with sufficient confidence.")
