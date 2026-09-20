"""
Erdos problem #614 -- quantum-testable-sequence lane.

Source record (erdosproblems.com data, /home/user/manman4/erdosproblems/data/
problems.yaml, entry "number: \"614\""):
    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["possible"]
    tags: ["graph theory"]

HONESTY NOTE / LIMITATION (read before trusting the PASS below):
Problem #614 carries NO real OEIS sequence id. The "oeis" field in the data
file literally contains the placeholder string "possible", not an actual
A-number, and the "tags" field carries only the single word "graph theory"
with no further formal statement extractable from this metadata file alone.
That means there is no genuine finite/computable *sequence* membership,
divisibility, or counting property of an OEIS sequence to derive from this
problem's own record -- the stated requirement ("identify a small, finite,
computable property ... from its OEIS sequence id(s) and tags") cannot be
honestly satisfied for #614, because the input needed (a real OEIS id) does
not exist in the source data.

Rather than fabricate an OEIS id or invent an unrelated numeric property and
pass it off as "the sequence for problem 614", this script is honest about
that gap and falls back to the closest defensible thing: it builds a REAL,
correct Grover search circuit over a small, self-contained, classically
verified graph-theory decision problem in the same spirit as the "graph
theory" tag (existence of a 2-colouring of the edges of K4, the complete
graph on 4 vertices, with no monochromatic triangle). This is a genuine
finite computable property with a real classical answer computed from first
principles below (not copied from any table), and the quantum circuit
genuinely searches for it -- but it is NOT claimed to be "the sequence for
Erdos problem 614", because no such finite computable OEIS-linked sequence
property could be extracted from the source record. ran_ok/verified_against
classical reported at the end describe this fallback instance honestly.

Classical problem instance actually solved:
    K4 has 6 edges. A 2-colouring assigns each edge one of 2 colours, i.e.
    a bitstring of length 6 (bit i = colour of edge i, 0 or 1). We search
    over all 2^6 = 64 colourings for one with NO monochromatic triangle
    (a triangle in K4 all of whose 3 edges share the same colour). K4 has
    4 triangles (choose 3 of 4 vertices). This is computed by brute force
    in Python first (ground truth), then a Grover search circuit is built
    that marks exactly the "good" (triangle-free-colouring) computational
    basis states among the 64, and is run on AerSimulator; the most
    frequent measured bitstring must be one of the classically-verified
    good states.

Qubits used: 6 (one per edge of K4). Oracle: computed by first building a
classical truth table (marking the "good" indices), which is itself the
ground truth used for verification, then compiling that exact set of
64 basis states into a diffusion-based Grover oracle via a multi-controlled
phase flip per marked state (small search space, so this is tractable and
exact -- no approximation).
"""

import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate


# ---------------------------------------------------------------------------
# 1. Classical ground truth, derived from first principles.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges of K4, indices 0..5
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles of K4
assert len(TRIANGLES) == 4


def triangle_edge_indices(tri):
    a, b, c = tri
    pairs = [(a, b), (a, c), (b, c)]
    idxs = []
    for p in pairs:
        p = tuple(sorted(p))
        idxs.append(EDGE_INDEX[p])
    return idxs


TRIANGLE_EDGE_IDXS = [triangle_edge_indices(t) for t in TRIANGLES]


def is_triangle_free_colouring(bits):
    """bits: tuple of 6 ints (0/1), one colour per edge, edge order = EDGES.
    Returns True iff no triangle of K4 is monochromatic."""
    for tri_edges in TRIANGLE_EDGE_IDXS:
        colours = {bits[i] for i in tri_edges}
        if len(colours) == 1:
            return False
    return True


good_states = []  # list of 6-bit tuples with no monochromatic triangle
for bits in itertools.product([0, 1], repeat=6):
    if is_triangle_free_colouring(bits):
        good_states.append(bits)

N = 64  # 2^6 search space size
M = len(good_states)
assert M > 0, "expected at least one triangle-free 2-colouring of K4"

# classical answer, printed for the record
good_bitstrings = ["".join(str(b) for b in reversed(bits)) for bits in good_states]
print(f"Classical brute force over all {N} edge-2-colourings of K4:")
print(f"  triangle-free colourings found: {M} of {N}")
print(f"  example good colouring (qiskit little-endian bitstring): {good_bitstrings[0]}")


# ---------------------------------------------------------------------------
# 2. Grover search circuit that marks exactly the good_states basis states.
# ---------------------------------------------------------------------------

NUM_QUBITS = 6


def build_oracle(marked_states):
    """Phase-flip oracle: multi-controlled Z (with appropriate X-sandwiching)
    on each marked computational basis state, in little-endian qubit order
    (qubit i corresponds to bits[i], edge i)."""
    qc = QuantumCircuit(NUM_QUBITS, name="oracle")
    mcz = MCMTGate(ZGate(), NUM_QUBITS - 1, 1)
    for bits in marked_states:
        # bits[i] is the colour of edge i -> mapped to qubit i (little endian)
        zero_positions = [i for i in range(NUM_QUBITS) if bits[i] == 0]
        if zero_positions:
            qc.x(zero_positions)
        qc.append(mcz, list(range(NUM_QUBITS)))
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    mcz = MCMTGate(ZGate(), n - 1, 1)
    qc.append(mcz, list(range(n)))
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(good_states)
diffuser = build_diffuser(NUM_QUBITS)

# optimal number of Grover iterations for N states, M marked
num_iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(num_iterations):
    qc.append(oracle.to_instruction(), range(NUM_QUBITS))
    qc.append(diffuser.to_instruction(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical ground truth.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
qc_decomposed = qc.decompose(reps=3)
result = sim.run(qc_decomposed, shots=shots).result()
counts = result.get_counts()

good_bitstring_set = set(good_bitstrings)
top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])

# fraction of shots landing on a classically-verified good state
good_shots = sum(c for bs, c in counts.items() if bs in good_bitstring_set)
good_fraction = good_shots / shots

print(f"\nGrover search used {num_iterations} iteration(s) over {NUM_QUBITS} qubits.")
print(f"Most frequent measured bitstring: {top_bitstring} (count {top_count}/{shots})")
print(f"Fraction of all shots landing on a classically-verified good state: {good_fraction:.3f}")

quantum_found_good = top_bitstring in good_bitstring_set
majority_good = good_fraction > 0.5

verified = quantum_found_good and majority_good

if verified:
    print("\nPASS: Grover search's top measurement is a classically-verified "
          "triangle-free K4 edge-2-colouring, and the majority of shots "
          "amplified onto the marked (good) subspace.")
else:
    print("\nFAIL: quantum result did not match the classical ground truth.")

print(
    "\nReminder: this circuit verifies a self-contained graph-theory search "
    "problem inspired by problem 614's 'graph theory' tag. It is NOT a "
    "verification of any OEIS sequence for problem 614, because the source "
    "record's oeis field for 614 is the placeholder 'possible', not a real "
    "OEIS id -- see the module docstring for the full honesty note."
)
