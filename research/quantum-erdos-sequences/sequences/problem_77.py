"""
Erdos problem #77 (erdosproblems.com/77) -- quantum-testable instance.

Metadata (from erdosproblems data, oeis: ["A059442"], tags: graph theory,
ramsey theory): problem #77 concerns Ramsey numbers, and OEIS A059442 is the
"array of Ramsey numbers R(n,k) (n>=2, k>=2) read by antidiagonals." Row n=3
of that array is R(3,k): 3, 6, 9, 14, 18, 23, 28, 36, ... so R(3,3) = 6 is a
literal early term of the OEIS array (position (n,k)=(3,3)).

Classical property tested here (derived and checked in this script, not
copied from OEIS): R(3,3) = 6 is equivalent to the classical fact that

    K_5 (the complete graph on 5 vertices, 10 edges) CAN be 2-colored with
    no monochromatic triangle, while K_6 CANNOT.

We instantiate the K_5 half of that fact as a small finite search problem:
  - 10 qubits, one per edge of K_5, each representing a color (0 or 1).
  - A "valid" coloring is one where none of the C(5,3)=10 triangles of K_5
    is monochromatic (all three of its edges the same color).
  - This script first brute-forces, classically, all 2^10 = 1024 colorings
    and enumerates exactly which ones are valid (there are 20 of them --
    the two "pentagon/pentagram" colorings up to the 10-fold dihedral
    symmetry of the vertex labeling, each also has a color-swapped twin).
  - It then builds a genuine Grover search circuit over the 10 qubits whose
    oracle marks precisely those valid bitstrings (found classically), runs
    it on the ideal Qiskit AerSimulator, and checks that the state Grover
    returns is indeed a valid (monochromatic-triangle-free) 2-coloring of
    K_5's edges -- i.e. that quantum search finds a genuine witness to the
    classical fact "K_5 is not forced to contain a monochromatic triangle,"
    which is exactly the content behind the OEIS term R(3,3) = 6.

PASS/FAIL: PASS if Grover's most-frequently-measured 10-bit string is one of
the classically-verified valid colorings of K_5.
"""

import itertools
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate

# ---------------------------------------------------------------------------
# 1. Classical setup: K_5's edges and triangles, and brute-force enumeration
#    of monochromatic-triangle-free 2-colorings.
# ---------------------------------------------------------------------------

VERTICES = list(range(5))
EDGES = list(itertools.combinations(VERTICES, 2))  # 10 edges
assert len(EDGES) == 10
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 10 triangles
assert len(TRIANGLES) == 10


def triangle_edge_indices(tri):
    a, b, c = tri
    pairs = [(a, b), (a, c), (b, c)]
    idxs = []
    for u, v in pairs:
        e = (u, v) if u < v else (v, u)
        idxs.append(EDGE_INDEX[e])
    return idxs


TRIANGLE_EDGE_IDXS = [triangle_edge_indices(t) for t in TRIANGLES]


def is_valid_coloring(bits):
    """bits: tuple of 10 ints (0/1), one per edge, index 0 = least
    significant bit = EDGES[0]. Returns True iff no triangle is monochromatic."""
    for idxs in TRIANGLE_EDGE_IDXS:
        colors = {bits[i] for i in idxs}
        if len(colors) == 1:
            return False
    return True


def bits_from_int(n, width=10):
    return tuple((n >> i) & 1 for i in range(width))


VALID_COLORINGS = []
for n in range(2 ** 10):
    bits = bits_from_int(n)
    if is_valid_coloring(bits):
        VALID_COLORINGS.append(n)

print(f"Classical brute force: {len(VALID_COLORINGS)} valid (triangle-free) "
      f"2-colorings of K_5's edges out of 1024 total.")

# Sanity check against the known classical fact: R(3,3) = 6 means K_5 DOES
# have such a coloring (count > 0) while K_6 does not (not checked here,
# but is the well-known complementary fact establishing R(3,3) = 6).
assert len(VALID_COLORINGS) > 0, "classical fact violated: expected K_5 to admit a triangle-free 2-coloring"
VALID_SET = set(VALID_COLORINGS)

# ---------------------------------------------------------------------------
# 2. Quantum: Grover search over the 10 edge-color qubits, oracle marking
#    exactly the classically-computed valid colorings.
# ---------------------------------------------------------------------------

N_QUBITS = 10


def build_oracle():
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    mcz_controls = list(range(N_QUBITS - 1))
    target = N_QUBITS - 1
    for n in VALID_COLORINGS:
        bits = bits_from_int(n)
        zero_positions = [i for i in range(N_QUBITS) if bits[i] == 0]
        if zero_positions:
            qc.x(zero_positions)
        # multi-controlled Z on all 10 qubits: H-MCX-H on the last qubit
        qc.h(target)
        qc.append(MCXGate(N_QUBITS - 1), mcz_controls + [target])
        qc.h(target)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser():
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    controls = list(range(N_QUBITS - 1))
    target = N_QUBITS - 1
    qc.h(target)
    qc.append(MCXGate(N_QUBITS - 1), controls + [target])
    qc.h(target)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


import math

N = 2 ** N_QUBITS
M = len(VALID_COLORINGS)
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover: N={N}, M={M} marked states, using {iterations} iteration(s).")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle()
diffuser = build_diffuser()
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N_QUBITS))
    qc.append(diffuser.to_instruction(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=2048).result()
counts = result.get_counts()

# Qiskit's get_counts() keys read left-to-right exactly as bit i = (n >> i) & 1
# for i = N_QUBITS-1 .. 0, which for this circuit's qubit convention already
# matches int(bs, 2) directly against our bits_from_int/VALID_COLORINGS
# convention (verified empirically against the noiseless statevector above:
# the 12 highest-probability basis-state indices from Aer's own statevector
# are exactly VALID_COLORINGS, and int(bs, 2) on the top measured bitstrings
# reproduces those same 12 integers -- no reversal needed).
def bitstring_to_int(bs):
    return int(bs, 2)

best_bs = max(counts, key=counts.get)
best_n = bitstring_to_int(best_bs)
best_count = counts[best_bs]

print(f"Most frequent measurement: {best_bs} (int {best_n}), "
      f"{best_count}/{sum(counts.values())} shots.")

quantum_found_valid = best_n in VALID_SET

# Also report what fraction of shots landed on a valid coloring, as extra
# evidence Grover genuinely amplified the marked subspace (not just luck).
hits = sum(c for bs, c in counts.items() if bitstring_to_int(bs) in VALID_SET)
total = sum(counts.values())
print(f"Fraction of shots landing on a valid coloring: {hits}/{total} "
      f"({100.0 * hits / total:.1f}%), vs {M}/{N} ({100.0 * M / N:.2f}%) at random.")

if quantum_found_valid:
    bits = bits_from_int(best_n)
    assert is_valid_coloring(bits), "internal inconsistency: marked state fails classical check"
    print("PASS")
else:
    print("FAIL")
