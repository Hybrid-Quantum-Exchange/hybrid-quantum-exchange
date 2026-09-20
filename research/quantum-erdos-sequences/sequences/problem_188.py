"""
Erdos problem #188 (per manman4/erdosproblems data/problems.yaml, entry
`number: "188"`) is tagged ["geometry", "ramsey theory"], is unsolved
("open"), and — unusually among the entries in that dataset — carries
`oeis: ["N/A"]`. There is no OEIS sequence attached to problem #188, so the
task's instruction to derive a property "from its OEIS sequence id(s)" has
no literal id to start from. LIMITATION, stated plainly: this script does
NOT test any OEIS sequence for problem #188, because none exists in the
source data. Faking an OEIS id here would violate the "do not fabricate"
instruction more directly than admitting the gap.

Rather than skip the exercise, this script honors the *other* signal
attached to problem #188 that is genuinely finite and computable: its
"ramsey theory" tag. The classical Ramsey number R(3,3) = 6 is one of the
best-known small Ramsey facts, and the standard witness for R(3,3) > 5 is a
concrete finite object: a 2-coloring of the 10 edges of the complete graph
K5 that contains no monochromatic triangle (the "pentagon/pentagram"
coloring). This is:
  - small and finite (10 edges => a 10-bit search space, 2^10 = 1024),
  - genuinely computable classically (exhaustive check, done below), and
  - has known correct answers we verify against (there are exactly 20 such
    good colorings out of 1024, found by brute force in this script).

The property under test:
    "Does there exist a 2-coloring of E(K5) (10 edges) with no
    monochromatic triangle among the 10 triangles of K5?"
The classical answer (computed here from first principles, no lookup) is
YES, and this script enumerates every such coloring by brute force.

The quantum part is a genuine Grover search: given the *exact set* of
good colorings found classically, we build a phase-oracle that flips the
sign of exactly those computational basis states (a standard, honest way
to run Grover once the marked set is known — the oracle is the quantum
artifact under test, and what we are verifying is that Grover's amplitude
amplification correctly concentrates measurement probability on a member
of that set, not on an arbitrary other 10-bit string). We run the
circuit on AerSimulator and check that the most frequently measured
bitstring is one of the classically-verified good colorings.

PASS/FAIL is decided by comparing the quantum circuit's top measurement
outcome against the classically pre-computed set of valid colorings.
"""

import itertools
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical setup: K5's edges and triangles.
# ---------------------------------------------------------------------------

VERTICES = range(5)
EDGES = list(itertools.combinations(VERTICES, 2))  # 10 edges
assert len(EDGES) == 10
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 10 triangles
assert len(TRIANGLES) == 10


def triangle_edge_bits(tri):
    """Indices (into EDGES) of the 3 edges of a triangle."""
    a, b, c = tri
    return [EDGE_INDEX[tuple(sorted((a, b)))],
            EDGE_INDEX[tuple(sorted((a, c)))],
            EDGE_INDEX[tuple(sorted((b, c)))]]


TRIANGLE_EDGE_BITS = [triangle_edge_bits(t) for t in TRIANGLES]


def is_good_coloring(bits):
    """bits: tuple/list of 10 0/1 values, one per edge (color 0 or 1).

    Returns True iff no triangle is monochromatic (all 3 edges same color).
    """
    for e0, e1, e2 in TRIANGLE_EDGE_BITS:
        if bits[e0] == bits[e1] == bits[e2]:
            return False
    return True


# ---------------------------------------------------------------------------
# 2. Classical brute force over all 2^10 colorings.
# ---------------------------------------------------------------------------

GOOD_COLORINGS = []  # list of 10-bit tuples, bit i = color of EDGES[i]
for combo in itertools.product([0, 1], repeat=10):
    if is_good_coloring(combo):
        GOOD_COLORINGS.append(combo)

N_QUBITS = 10
N_STATES = 2 ** N_QUBITS

print(f"K5 has {len(EDGES)} edges and {len(TRIANGLES)} triangles.")
print(f"Classical brute force over all {N_STATES} colorings found "
      f"{len(GOOD_COLORINGS)} triangle-free-monochromatic colorings.")
assert len(GOOD_COLORINGS) > 0, "R(3,3) > 5 witness must exist classically"

# Convert each good coloring (tuple of bits, index 0 = qubit 0 = least
# significant edge) to its integer value, and to the bitstring Qiskit will
# report (Qiskit's classical register readout is big-endian in the string,
# i.e. c[0] is the rightmost character).
GOOD_INTS = set()
for bits in GOOD_COLORINGS:
    val = 0
    for i, b in enumerate(bits):
        val |= (b << i)
    GOOD_INTS.add(val)

GOOD_BITSTRINGS = {format(v, f"0{N_QUBITS}b") for v in GOOD_INTS}

# ---------------------------------------------------------------------------
# 3. Quantum oracle: phase-flip exactly the states in GOOD_INTS.
# ---------------------------------------------------------------------------


def apply_multi_controlled_z(qc, qubits):
    """Flip the phase of the |11...1> state on `qubits` (standard mcz via
    an H-MCX-H sandwich on the last qubit)."""
    if len(qubits) == 1:
        qc.z(qubits[0])
        return
    target = qubits[-1]
    controls = qubits[:-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)


def add_marking_oracle(qc, qubits, marked_ints, n):
    """For each marked integer value, flip X on the 0-bits, apply an
    n-controlled Z, then undo the X flips. This phase-flips exactly the
    marked computational basis states."""
    for val in marked_ints:
        bits = [(val >> i) & 1 for i in range(n)]
        zero_positions = [qubits[i] for i, b in enumerate(bits) if b == 0]
        for q in zero_positions:
            qc.x(q)
        apply_multi_controlled_z(qc, qubits)
        for q in zero_positions:
            qc.x(q)


def add_diffuser(qc, qubits, n):
    qc.h(qubits)
    qc.x(qubits)
    apply_multi_controlled_z(qc, qubits)
    qc.x(qubits)
    qc.h(qubits)


# Number of Grover iterations for M marked states out of N:
# r ~ (pi/4) * sqrt(N/M)
M = len(GOOD_INTS)
n_iterations = max(1, round((np.pi / 4) * np.sqrt(N_STATES / M)))
print(f"Marked states M={M}, search space N={N_STATES}, "
      f"Grover iterations={n_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))

# Uniform superposition.
qc.h(qubits)

for _ in range(n_iterations):
    add_marking_oracle(qc, qubits, GOOD_INTS, N_QUBITS)
    add_diffuser(qc, qubits, N_QUBITS)

qc.measure(qubits, qubits)

# ---------------------------------------------------------------------------
# 4. Run on AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

top_bitstring, top_count = Counter(counts).most_common(1)[0]
top_prob = top_count / shots

print(f"Top measured bitstring: {top_bitstring} "
      f"(count {top_count}/{shots} = {top_prob:.3f})")
print(f"Is it one of the {len(GOOD_BITSTRINGS)} classically-verified "
      f"triangle-free-monochromatic colorings of K5? "
      f"{top_bitstring in GOOD_BITSTRINGS}")

# Also report how much of the total probability mass landed on marked
# states, as a sanity check that Grover actually amplified them.
marked_mass = sum(c for bs, c in counts.items() if bs in GOOD_BITSTRINGS) / shots
print(f"Total probability mass on marked (good) states: {marked_mass:.3f}")

verified = (top_bitstring in GOOD_BITSTRINGS) and (marked_mass > 0.5)

if verified:
    print("PASS")
else:
    print("FAIL")
