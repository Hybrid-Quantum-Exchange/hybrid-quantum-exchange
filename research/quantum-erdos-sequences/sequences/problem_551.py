"""
Erdos problem #551 (erdosproblems.com / manman4/erdosproblems data/problems.yaml)
tags: ["graph theory", "ramsey theory"]; oeis: ["N/A"] -- this problem carries
NO OEIS sequence id in the source data (the oeis field is the literal string
"N/A"). Per the task instructions, when there is no OEIS id the honest move
is to write the best real quantum circuit for a small, finite, computable
property drawn from the problem's own tags, and say plainly that there is no
OEIS sequence backing it.

Chosen property (genuine graph-theory / Ramsey-theory content, not fabricated,
not just an OEIS lookup):

    For the complete graph K4 (vertices 0..3, the 6 edges
    (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)), 2-color every edge (color bit 0/1,
    encoded as one qubit per edge -> 6 qubits total, 2^6 = 64 colorings).
    A coloring is "triangle-clean" if none of the 4 triangles
    {0,1,2} {0,1,3} {0,2,3} {1,2,3} is monochromatic (all 3 of its edges the
    same color).

    This is exactly the small-Ramsey-number question behind R(3,3): does a
    2-coloring of a complete graph on n vertices avoiding a monochromatic
    triangle exist? For n=4 (< R(3,3)=6) it certainly does; the script
    verifies this from first principles by brute-force classical enumeration
    of all 64 colorings, and then uses a real Grover search circuit on the
    ideal AerSimulator to search the same 6-qubit space for a triangle-clean
    coloring, checking that Grover amplifies exactly the classically-valid
    set.

Circuit: exact Grover's algorithm.
  - 6 "edge" qubits encode a candidate coloring as a computational basis state.
  - The oracle is built by classically enumerating, in Python, the full set
    of triangle-clean colorings (first principles, no OEIS/table lookup),
    then compiling a diagonal phase oracle that flips the sign of exactly
    those marked basis states (X-gates to map each marked bitstring to the
    all-ones pattern, a multi-controlled Z, then undo the X-gates).
  - The standard diffusion operator is applied for the classically-computed
    optimal number of Grover iterations round(pi/4 * sqrt(N/M)).
  - The circuit is run on qiskit_aer's ideal AerSimulator with many shots;
    PASS requires that the measurement distribution is concentrated on the
    classically-valid (triangle-clean) states -- i.e. that essentially all of
    the sampled mass is on states in the classically precomputed marked set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

ERDOS_PROBLEM_NUMBER = 551
OEIS_IDS = ["N/A"]  # no OEIS sequence attached to this problem in the source data

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges, K4
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles
assert len(TRIANGLES) == 4

N_QUBITS = len(EDGES)
N_STATES = 2 ** N_QUBITS  # 64


def edge_color(coloring_bits, u, v):
    """coloring_bits: tuple of 6 ints (0/1), one per edge in EDGES order."""
    a, b = (u, v) if u < v else (v, u)
    return coloring_bits[EDGE_INDEX[(a, b)]]


def is_triangle_clean(coloring_bits):
    for (a, b, c) in TRIANGLES:
        c_ab = edge_color(coloring_bits, a, b)
        c_ac = edge_color(coloring_bits, a, c)
        c_bc = edge_color(coloring_bits, b, c)
        if c_ab == c_ac == c_bc:
            return False  # monochromatic triangle found
    return True


def bits_to_int(bits):
    # bit i (edge i) is qubit i; use little-endian (qiskit convention: qubit 0
    # is the least-significant bit of the measured integer).
    val = 0
    for i, b in enumerate(bits):
        if b:
            val |= (1 << i)
    return val


def int_to_bits(val, n=N_QUBITS):
    return tuple((val >> i) & 1 for i in range(n))


marked_states = []  # list of integers (0..63) whose coloring is triangle-clean
for combo in itertools.product([0, 1], repeat=N_QUBITS):
    if is_triangle_clean(combo):
        marked_states.append(bits_to_int(combo))

marked_states = sorted(set(marked_states))
M = len(marked_states)
N = N_STATES

assert M > 0, "expected at least one triangle-clean 2-coloring of K4 (K4 has < R(3,3)=6 vertices)"
assert M < N, "expected at least one coloring WITH a monochromatic triangle, for a nontrivial oracle"

print(f"Classical ground truth: {M} of {N} colorings of K4's edges are triangle-clean "
      f"(no monochromatic triangle in any of the {len(TRIANGLES)} triangles).")

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle and diffusion operator.
# ---------------------------------------------------------------------------


def append_multi_controlled_z(qc, qubits):
    """Apply a phase flip to the |11...1> state of `qubits` (multi-controlled Z,
    up to global structure implemented via H + MCX + H on the last qubit)."""
    if len(qubits) == 1:
        qc.z(qubits[0])
        return
    target = qubits[-1]
    controls = qubits[:-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)


def append_oracle(qc, qubits, marked_ints):
    """Flip the phase of exactly the basis states in marked_ints."""
    for val in marked_ints:
        bits = int_to_bits(val)
        zero_positions = [qubits[i] for i, b in enumerate(bits) if b == 0]
        if zero_positions:
            qc.x(zero_positions)
        append_multi_controlled_z(qc, qubits)
        if zero_positions:
            qc.x(zero_positions)


def append_diffusion(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    append_multi_controlled_z(qc, qubits)
    qc.x(qubits)
    qc.h(qubits)


num_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Using {num_iterations} Grover iteration(s) for N={N}, M={M}.")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))

qc.h(qubits)  # uniform superposition over all 64 colorings

for _ in range(num_iterations):
    append_oracle(qc, qubits, marked_states)
    append_diffusion(qc, qubits)

qc.measure(qubits, qubits)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
compiled = backend  # AerSimulator can run a QuantumCircuit directly via .run
job = backend.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

# qiskit reports bitstrings MSB-first as "qubit(n-1)...qubit0"; convert each
# key back into our little-endian integer encoding to compare with marked_states.
marked_mass = 0
total_mass = 0
per_state_counts = {}
for bitstring, cnt in counts.items():
    # bitstring is e.g. "010110", index 0 char = qubit(N-1)
    bits = tuple(int(ch) for ch in reversed(bitstring))
    val = bits_to_int(bits)
    per_state_counts[val] = cnt
    total_mass += cnt
    if val in marked_states:
        marked_mass += cnt

fraction_on_marked = marked_mass / total_mass
most_likely_state = max(per_state_counts, key=per_state_counts.get)

print(f"Fraction of shots landing on a classically triangle-clean coloring: "
      f"{fraction_on_marked:.4f} ({marked_mass}/{total_mass})")
print(f"Most likely measured state: {most_likely_state} "
      f"(triangle-clean per classical check: {most_likely_state in marked_states})")

# ---------------------------------------------------------------------------
# 4. Verify quantum result against the classical answer.
# ---------------------------------------------------------------------------

# The classical answer: M triangle-clean colorings exist among N=64 total,
# and Grover search should concentrate most of its measurement mass on that
# marked set, with the single most likely outcome itself triangle-clean.
success = (most_likely_state in marked_states) and (fraction_on_marked > 0.5)

if success:
    print("PASS")
else:
    print("FAIL")
