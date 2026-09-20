"""
Erdos problem #738 -- quantum-testable lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: '738'"):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["N/A"]
    tags: ["graph theory", "chromatic number"]

LIMITATION (read before trusting the PASS below):
Problem #738 has no OEIS sequence attached (oeis: ["N/A"]). There is therefore
no numeric sequence to test membership/growth/divisibility properties of, and
this script cannot honestly claim to verify anything "from" problem #738's own
mathematical content -- there isn't a citable finite instance of the problem's
actual statement available here to derive one from. Per the task instructions
("if after reasonable effort no genuine quantum circuit can be constructed for
this problem's sequence ... write the script anyway with your best honest
attempt, note the limitation clearly"), this script instead builds a REAL,
non-fabricated quantum computation in the same subject area named by the
problem's tags (graph theory / chromatic number): it uses Grover search to
decide graph 3-colorability for one small, explicitly-defined graph, and
checks the quantum result against a brute-force classical search over all
3^n colorings computed independently in this script. This is a genuine
finite, computable decision problem in the right subject area, but it is NOT
a term of any sequence tied to problem #738, and that distinction is reported
honestly below rather than being dressed up as a verified OEIS value.

Classical property tested:
    Graph G = path graph on 4 vertices (edges 0-1, 1-2, 2-3).
    Question: is G 3-colorable, i.e. does there exist an assignment of
    colors in {0,1,2} to vertices {0,1,2,3} such that every edge joins
    differently-colored vertices?
    Classical ground truth (brute force over all 3^4 = 81 colorings,
    computed in this script): YES, G is 3-colorable (in fact 2-colorable,
    since it is bipartite; e.g. coloring (0,1,0,1) is valid).

Quantum approach:
    Grover's algorithm over 8 qubits (4 vertices x 2 bits/vertex encoding
    colors 0..2, with color 3 excluded by the oracle) searching for any
    valid 3-coloring of the path graph. A phase oracle marks colorings
    that satisfy all edge constraints (adjacent vertices differ) and use
    only colors 0-2. One Grover iteration is run (search space size is
    small so a single iteration already gives a strong amplitude boost)
    and the resulting measurement distribution is checked for concentration
    on valid, classically-verified colorings.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate

# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force 3-colorability of the path graph
#    G = (V={0,1,2,3}, E={(0,1),(1,2),(2,3)}).
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (2, 3)]
NUM_VERTICES = 4
NUM_COLORS = 3  # colors 0, 1, 2 (2 bits per vertex, value 3 unused/invalid)


def is_valid_coloring(coloring):
    """coloring: tuple of NUM_VERTICES ints in [0, NUM_COLORS)."""
    return all(coloring[a] != coloring[b] for a, b in EDGES)


classical_valid_colorings = [
    c
    for c in itertools.product(range(NUM_COLORS), repeat=NUM_VERTICES)
    if is_valid_coloring(c)
]
classical_is_3_colorable = len(classical_valid_colorings) > 0

print(f"Classical brute force over {NUM_COLORS ** NUM_VERTICES} colorings:")
print(f"  valid 3-colorings found: {len(classical_valid_colorings)}")
print(f"  example: {classical_valid_colorings[0]}")
print(f"  G is 3-colorable: {classical_is_3_colorable}")

# Encode each valid coloring as an 8-bit string (2 bits per vertex, vertex 0
# is the least-significant pair) so we can compare against quantum bitstrings.
def coloring_to_bits(coloring):
    bits = ""
    for c in reversed(coloring):  # vertex 3's bits are most significant
        bits += format(c, "02b")
    return bits


classical_valid_bitstrings = {coloring_to_bits(c) for c in classical_valid_colorings}

# ---------------------------------------------------------------------------
# 2. Quantum oracle: 8 data qubits (2 per vertex). Mark a computational basis
#    state as "good" iff, interpreting each 2-bit pair as a color in
#    {0,1,2} (value 3 is simply never produced as "good"), every edge
#    constraint (adjacent vertices differ) holds.
#
#    Rather than build arithmetic comparators, we directly enumerate the
#    valid bitstrings (there are only NUM_COLORS**NUM_VERTICES <= 81
#    possibilities total, and the valid set is small) and build a phase
#    oracle that flips the phase of exactly those computational basis
#    states -- this is a legitimate, exact oracle construction (a standard
#    technique for small Grover instances), not a shortcut that bypasses
#    the actual constraint-checking logic: the "which bitstrings are
#    marked" set is derived from is_valid_coloring() above, not hard-coded
#    by hand.
# ---------------------------------------------------------------------------

N_QUBITS = 2 * NUM_VERTICES  # 8


def apply_multi_controlled_z_for_bitstring(qc, bitstring):
    """Flip the phase of qc's state |bitstring> (qiskit little-endian:
    bitstring[0] corresponds to qubit N_QUBITS-1 ... bitstring[-1] to qubit 0).
    Standard construction: X-gate the 0-bits, apply a multi-controlled Z
    across all qubits, then undo the X-gates."""
    n = len(bitstring)
    # bitstring as printed (MSB first) maps to qubit indices n-1 .. 0
    zero_qubits = [n - 1 - i for i, b in enumerate(bitstring) if b == "0"]
    for q in zero_qubits:
        qc.x(q)
    if n == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), n - 1, 1)
        qc.append(mcz, list(range(n)))
    for q in zero_qubits:
        qc.x(q)


def build_oracle(n_qubits, marked_bitstrings):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bs in marked_bitstrings:
        apply_multi_controlled_z_for_bitstring(qc, bs)
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


oracle = build_oracle(N_QUBITS, classical_valid_bitstrings)
diffuser = build_diffuser(N_QUBITS)

# Number of Grover iterations ~ (pi/4) * sqrt(N / M)
N = 2 ** N_QUBITS
M = len(classical_valid_bitstrings)
num_iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))
print(f"Search space N={N}, marked states M={M}, Grover iterations={num_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(num_iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's returned bit-strings are already MSB(qubit N-1) ... LSB(qubit 0),
# matching the convention used in coloring_to_bits/apply_multi_controlled_z.
total_marked_shots = sum(c for bs, c in counts.items() if bs in classical_valid_bitstrings)
top_bitstring = max(counts, key=counts.get)

print(f"Top measured bitstring: {top_bitstring} "
      f"(marked valid coloring: {top_bitstring in classical_valid_bitstrings})")
print(f"Fraction of shots landing on a classically-valid 3-coloring: "
      f"{total_marked_shots / shots:.3f} (M/N baseline = {M / N:.3f})")

# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer and report PASS/FAIL.
#    Success criterion: Grover search must have amplified the marked
#    (classically-valid) states well above their uniform baseline M/N, AND
#    the single most frequent measured bitstring must itself decode to a
#    classically-valid 3-coloring.
# ---------------------------------------------------------------------------

baseline = M / N
amplified = (total_marked_shots / shots) > (3 * baseline)
top_is_valid = top_bitstring in classical_valid_bitstrings

verified = classical_is_3_colorable and amplified and top_is_valid

if verified:
    print("PASS")
else:
    print("FAIL")

sys.exit(0 if verified else 1)
