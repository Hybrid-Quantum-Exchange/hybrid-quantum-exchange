"""
Erdos problem #598 (https://www.erdosproblems.com/598), quantum-testable lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: '598'"):
    prize: no
    informal_status: open (last update 2025-08-31)
    oeis: ["N/A"]
    tags: ["set theory", "ramsey theory"]

LIMITATION, stated honestly up front: problem #598 carries no OEIS sequence id
("N/A") and its statement is a set-theory/Ramsey-theory question that is not
itself a finite computable sequence membership test. There is therefore no
literal OEIS term to defer to. Per the task's fallback instructions, this
script instead builds a genuine, small, finite, computable instance of the
same *tag* the problem is filed under -- Ramsey theory -- and verifies it with
a real Grover search circuit, rather than fabricating an OEIS value that does
not exist for this problem.

Classical property tested (computed from first principles in this script, not
copied from anywhere):

    For the complete graph K4 (4 vertices, 6 edges), does there exist a
    2-coloring of the edges with NO monochromatic triangle? (This is exactly
    the finite question behind the classical Ramsey number R(3,3) = 6: for
    n < 6 vertices such "good" colorings exist; R(3,3) is itself a famous
    small Ramsey-theory fact of the same flavor as problem #598's tags.)

    K4 has 6 edges and C(4,3) = 4 triangles. Each edge is colored with one
    bit (0/1), giving a search space of 2^6 = 64 colorings. A coloring is
    "good" if none of the 4 triangles has all three of its edges the same
    color.

    This script first brute-forces, purely classically, the exact set of
    good colorings among all 64 bitstrings (18 of them, verified below).
    It then builds a Grover search circuit over the 6-qubit edge-coloring
    register whose oracle marks exactly that classically-computed good set,
    runs it on the ideal AerSimulator, and checks that the state the quantum
    search returns most often is indeed one of the classically-verified good
    colorings (and, as a stronger check, that the whole set of "loud" output
    states -- those measured above the uniform-noise floor -- is a subset of
    the classical good set). This is a genuine amplitude-amplification
    search (Grover's algorithm), not a lookup: the oracle only encodes the
    classical solution set as phase flips, and the diffusion operator does
    the amplification; the quantum computation is what steers the
    measurement distribution towards the good colorings from a uniform
    superposition over all 64 possibilities.

Report fields for this run: ran_ok / verified_against_classical are reported
based on the actual execution below.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (ground truth), from first principles.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # 6 edges of K4
TRIANGLES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]  # 4 triangles of K4


def edge_index(a, b):
    e = (a, b) if a < b else (b, a)
    return EDGES.index(e)


def is_good_coloring(bits):
    """bits: tuple of 6 ints (0/1), one per edge in EDGES order.
    Returns True iff no triangle in TRIANGLES is monochromatic."""
    for (a, b, c) in TRIANGLES:
        e_ab = bits[edge_index(a, b)]
        e_ac = bits[edge_index(a, c)]
        e_bc = bits[edge_index(b, c)]
        if e_ab == e_ac == e_bc:
            return False
    return True


GOOD = [bits for bits in itertools.product((0, 1), repeat=6) if is_good_coloring(bits)]
GOOD_SET = set(GOOD)

assert len(GOOD) == 18, f"expected 18 good colorings of K4, classically found {len(GOOD)}"
print(f"Classical brute force: {len(GOOD)} / 64 edge-colorings of K4 avoid a "
      f"monochromatic triangle (property behind R(3,3) = 6).")

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 6-qubit coloring register.
# ---------------------------------------------------------------------------

N_QUBITS = 6  # one qubit per edge; qiskit bit order is little-endian (qubit 0 = LSB)


def mark_state(qc, bits_msb_first):
    """Flip the phase of the single computational basis state described by
    bits_msb_first (a tuple of 6 ints, in EDGES order = same order used
    classically). Standard X / multi-controlled-Z / X sandwich."""
    # Qiskit qubit i corresponds to EDGES[i]; use that same ordering directly.
    zero_positions = [i for i, b in enumerate(bits_msb_first) if b == 0]
    for i in zero_positions:
        qc.x(i)
    # multi-controlled Z on all N_QUBITS-1 controls + 1 target, via H-MCX-H
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    for i in zero_positions:
        qc.x(i)


def build_oracle():
    qc = QuantumCircuit(N_QUBITS, name="oracle_good_colorings")
    for bits in GOOD:
        mark_state(qc, bits)
    return qc


def build_diffuser():
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


N = 2 ** N_QUBITS          # 64
M = len(GOOD)               # 18 marked states
# optimal number of Grover iterations for M marked items out of N
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover: N={N}, M={M} marked states, using {iterations} iteration(s).")

oracle = build_oracle()
diffuser = build_diffuser()

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
tqc = transpile(qc, sim)
SHOTS = 20000
result = sim.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

# Qiskit's bit string is c[N-1]...c[0]; c[i] is the measured value of qubit i,
# which we assigned to EDGES[i] above -- so reverse the printed string to get
# back to our (EDGES[0], ..., EDGES[5]) bit tuple.
def bitstring_to_tuple(bs):
    rev = bs[::-1]
    return tuple(int(c) for c in rev)

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_state, top_count = sorted_counts[0]
top_tuple = bitstring_to_tuple(top_state)

uniform_expected = SHOTS / N  # ~312.5 -- what a state would get with no amplification
loud_states = [bs for bs, c in sorted_counts if c > 3 * uniform_expected]
loud_tuples = [bitstring_to_tuple(bs) for bs in loud_states]

top_is_good = top_tuple in GOOD_SET
loud_all_good = all(t in GOOD_SET for t in loud_tuples) and len(loud_tuples) > 0

print(f"Top measured coloring: {top_tuple} with {top_count}/{SHOTS} shots "
      f"(classically good: {top_is_good}).")
print(f"Number of 'loud' (amplified) distinct states measured: {len(loud_tuples)}, "
      f"all classically good: {loud_all_good}.")

verified = bool(top_is_good and loud_all_good)

if verified:
    print("PASS")
else:
    print("FAIL")
