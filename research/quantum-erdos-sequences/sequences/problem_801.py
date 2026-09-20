"""
Erdos problem #801 (from manman4/erdosproblems, data/problems.yaml).

Metadata found in the source YAML for problem 801:
    number: "801"
    prize: "no"
    status: proved (last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["graph theory", "ramsey theory"]

LIMITATION: problem 801 carries no OEIS sequence id ("N/A"), so there is no
"quantum-testable sequence" to build a circuit against in the literal sense
the task asks for. Rather than fabricate an OEIS id or copy an unrelated
sequence, this script honors the problem's own tags (graph theory / Ramsey
theory) and tests a small, finite, genuinely computable property from that
area that is closely related to the kind of statement Ramsey-theory Erdos
problems make:

    Classical property under test
    ------------------------------
    Consider the complete graph K4 (6 edges, vertices 0,1,2,3) with each
    edge independently colored red (0) or blue (1). A coloring is "good" if
    none of the 4 triangles of K4 is monochromatic (all three of its edges
    the same color). This is exactly the small-n side of the two-colour
    Ramsey number R(3,3)=6: for n<6 vertices, monochromatic-triangle-free
    colorings exist, and this script both (a) computes classically, by
    brute force over all 2^6=64 colorings, the exact set of good colorings,
    and (b) uses Grover's algorithm on a genuine oracle circuit (built from
    elementary equality/AND gates over the 6 edge qubits, with a
    multi-controlled phase flip) to amplify exactly those good colorings,
    then samples the resulting state and checks that the most likely
    measured bitstring is indeed a monochromatic-triangle-free coloring,
    matching the classical brute-force set.

This is a real Grover search (oracle + diffusion, correct number of
iterations computed from the true count of marked states, verified against
AerSimulator statevector/sampling), not a literal OEIS lookup.

No OEIS id was used (problem 801 has none). ran_ok / verified_against_classical
are reported honestly based on what this script actually does below.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute force over all 2^6 edge colorings of K4.
# ---------------------------------------------------------------------------

# Edge indexing: e0=(0,1) e1=(0,2) e2=(0,3) e3=(1,2) e4=(1,3) e5=(2,3)
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
EDGE_INDEX = {frozenset(e): i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(range(4), 3))  # 4 triangles of K4


def triangle_edge_indices(tri):
    a, b, c = tri
    return (
        EDGE_INDEX[frozenset((a, b))],
        EDGE_INDEX[frozenset((b, c))],
        EDGE_INDEX[frozenset((a, c))],
    )


TRI_EDGES = [triangle_edge_indices(t) for t in TRIANGLES]


def is_good_coloring(bits):
    """bits: tuple of 6 ints (0/1), bits[i] is the color of EDGES[i]."""
    for i0, i1, i2 in TRI_EDGES:
        if bits[i0] == bits[i1] == bits[i2]:
            return False
    return True


all_colorings = list(itertools.product([0, 1], repeat=6))
good_colorings = [c for c in all_colorings if is_good_coloring(c)]
N = len(all_colorings)          # 64
M = len(good_colorings)         # number of marked ("good") states

assert N == 64
assert M > 0, "expected at least one monochromatic-triangle-free coloring of K4"
print(f"Classical brute force: {M} good colorings out of {N} total "
      f"(K4, 2-coloring, no monochromatic triangle).")

# ---------------------------------------------------------------------------
# 2. Grover oracle: mark the "good" colorings by phase.
# ---------------------------------------------------------------------------
# Registers:
#   e[0..5]   : the 6 edge-color qubits (the search register)
#   mono[0..3]: one ancilla per triangle, holds 1 iff that triangle is
#               monochromatic in the current basis state
#   tmp[0..1] : two reusable ancillas for the pairwise-equality computation
#
# For a triangle with edges (a,b,c):
#   eq1 = NOT(a XOR b)   -> 1 iff a==b
#   eq2 = NOT(b XOR c)   -> 1 iff b==c
#   mono = eq1 AND eq2   -> 1 iff a==b==c  (monochromatic)
# eq1/eq2 are then uncomputed (returned to |0>) so tmp can be reused for the
# next triangle; mono[i] is left set until after the phase flip, then
# uncomputed by re-running the same triangle computation (self-inverse
# except for the final CCX, which we undo explicitly).

e = QuantumRegister(6, "e")
mono = QuantumRegister(4, "mono")
tmp = QuantumRegister(2, "tmp")


def compute_triangle(qc, a, b, c, mono_q):
    eq1, eq2 = tmp[0], tmp[1]
    # eq1 = NOT(a xor b)
    qc.cx(a, eq1)
    qc.cx(b, eq1)
    qc.x(eq1)
    # eq2 = NOT(b xor c)
    qc.cx(b, eq2)
    qc.cx(c, eq2)
    qc.x(eq2)
    # mono = eq1 AND eq2
    qc.ccx(eq1, eq2, mono_q)
    # uncompute eq2, eq1 (reverse order, each step is its own inverse)
    qc.x(eq2)
    qc.cx(c, eq2)
    qc.cx(b, eq2)
    qc.x(eq1)
    qc.cx(b, eq1)
    qc.cx(a, eq1)


def build_oracle():
    qc = QuantumCircuit(e, mono, tmp, name="oracle")
    # 1) compute mono[i] for each triangle
    for i, (i0, i1, i2) in enumerate(TRI_EDGES):
        compute_triangle(qc, e[i0], e[i1], e[i2], mono[i])
    # 2) phase-flip iff ALL mono[i] == 0 (i.e. no triangle is monochromatic)
    qc.x(mono)  # flip so "all zero" becomes "all one"
    qc.h(mono[3])
    qc.mcx(list(mono[0:3]), mono[3])
    qc.h(mono[3])
    qc.x(mono)  # uncompute the flip
    # 3) uncompute mono[i] (each triangle's computation is fully reversible)
    for i, (i0, i1, i2) in enumerate(TRI_EDGES):
        compute_triangle(qc, e[i0], e[i1], e[i2], mono[i])
    return qc


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


oracle = build_oracle()
diffuser = build_diffuser(6)

iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations used: {iterations} (N={N}, M={M})")

qc = QuantumCircuit(e, mono, tmp)
qc.h(e)
for _ in range(iterations):
    qc.append(oracle.to_instruction(), list(e) + list(mono) + list(tmp))
    qc.append(diffuser.to_instruction(), list(e))

qc.measure_all()

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------
sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit's measure_all bitstring is "mono(4) tmp(2) e(6)"-order reversed
# per-register with a space between the ancilla block and e; ancillas were
# uncomputed back to 0, so we just need the e-register bits. Extract them
# robustly using the classical register / qubit ordering that Qiskit uses:
# rightmost characters correspond to qubit 0 upward across the whole circuit.
def extract_e_bits(bitstring):
    clean = bitstring.replace(" ", "")
    # Full register order (Qiskit big-endian print, qubit 0 = rightmost bit)
    # total qubits = 6 (e) + 4 (mono) + 2 (tmp) = 12, e are qubits 0..5
    full = clean[::-1]  # now index i corresponds to qubit i
    e_bits = tuple(int(full[i]) for i in range(6))
    return e_bits


sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_bitstring, top_count = sorted_counts[0]
top_edge_bits = extract_e_bits(top_bitstring)

# sanity: ancillas should be back to |0...0>
ancilla_bits = extract_e_bits(top_bitstring)  # unused placeholder, real check below
clean_top = top_bitstring.replace(" ", "")[::-1]
ancilla_all_zero = all(b == "0" for b in clean_top[6:12])

quantum_answer_is_good = is_good_coloring(top_edge_bits)

# Aggregate probability mass landing on classically-good colorings, as an
# additional cross-check that Grover genuinely amplified the marked set.
good_set = set(good_colorings)
good_mass = sum(c for bs, c in counts.items() if extract_e_bits(bs) in good_set)
good_fraction = good_mass / shots

print(f"Most frequent measured edge-coloring: {top_edge_bits} "
      f"(count {top_count}/{shots}), ancillas clean: {ancilla_all_zero}")
print(f"Classically verified as monochromatic-triangle-free: {quantum_answer_is_good}")
print(f"Fraction of shots landing on a good coloring: {good_fraction:.3f} "
      f"(baseline uniform would be {M/N:.3f})")

verified = quantum_answer_is_good and ancilla_all_zero and good_fraction > (M / N)

if verified:
    print("PASS")
else:
    print("FAIL")
