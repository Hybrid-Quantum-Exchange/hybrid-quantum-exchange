"""
Erdos problem #111 (erdosproblems.com/111) — quantum-testable lane.

Source metadata (data/problems.yaml, manman4/erdosproblems clone, entry
"number: '111'"): prize "no", status "open", tags
["graph theory", "chromatic number", "set theory"], oeis: ["N/A"].

LIMITATION, stated honestly: problem 111 has no associated OEIS sequence id
in the source data (oeis: ["N/A"]). Per the task instructions, in that case
this script does its best honest attempt at a property that is directly
derived from the problem's own listed tags ("graph theory", "chromatic
number") rather than from any OEIS term, since there is no OEIS id to check
a term of. It is NOT a test of a specific OEIS sequence membership/term —
there isn't one to test. It is a genuine, finite, computable chromatic-number
decision problem, verified classically from first principles and then
checked with a real Grover search circuit on AerSimulator.

Classical property under test
------------------------------
Graph: path graph P3 on vertices {0, 1, 2} with edges (0,1) and (1,2).
Question: is P3 2-colorable (i.e. is its chromatic number <= 2)? This is a
finite instance of exactly the "chromatic number" question problem 111 is
tagged with.

Each vertex gets 1 qubit (color in {0, 1}), so the search space is the 3-bit
strings b2 b1 b0 (bit i = color of vertex i), size 8. A coloring is valid
(proper) iff adjacent vertices get different colors: bit0 != bit1 and
bit1 != bit2.

The script first brute-forces this classically (first principles, no OEIS
lookup) to get the exact set of valid colorings and hence the classical
answer to "is P3 2-colorable?" and the exact number of solutions among the
8 possible assignments. It then builds a genuine Grover search circuit
(oracle + diffuser, iterated the standard optimal number of times for the
known solution count) that amplifies exactly those valid-coloring basis
states, runs it on AerSimulator, and checks that the circuit's most likely
measured outcomes are exactly the classically-verified valid colorings.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2]
EDGES = [(0, 1), (1, 2)]  # path graph P3
N_QUBITS = len(VERTICES)  # one qubit per vertex, 2 colors {0,1}


def is_proper_coloring(bits):
    """bits: tuple of 0/1 per vertex index; True iff every edge's endpoints differ."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


def classical_valid_colorings():
    valid = []
    for bits in itertools.product([0, 1], repeat=N_QUBITS):
        if is_proper_coloring(bits):
            valid.append(bits)
    return valid


VALID_COLORINGS = classical_valid_colorings()
CHROMATIC_NUMBER_LE_2 = len(VALID_COLORINGS) > 0
N_SOLUTIONS = len(VALID_COLORINGS)
SEARCH_SPACE_SIZE = 2 ** N_QUBITS

print("Classical brute force over all %d colorings of P3 with 2 colors:"
      % SEARCH_SPACE_SIZE)
for bits in itertools.product([0, 1], repeat=N_QUBITS):
    print("  vertices(0,1,2)=%s  proper=%s" % (bits, is_proper_coloring(bits)))
print("Valid (proper) 2-colorings found: %s" % (VALID_COLORINGS,))
print("=> P3 is 2-colorable (chromatic number <= 2): %s, with exactly %d "
      "valid colorings out of %d assignments"
      % (CHROMATIC_NUMBER_LE_2, N_SOLUTIONS, SEARCH_SPACE_SIZE))

assert N_SOLUTIONS == 2, "sanity check on the classical brute force itself"
assert CHROMATIC_NUMBER_LE_2 is True


# ---------------------------------------------------------------------------
# 2. Grover search circuit that finds a valid coloring.
#
# Qubit layout: q0 = color of vertex 0, q1 = color of vertex 1 (also the
# vertex we bit-flip against for the "not equal" checks), q2 = color of
# vertex 2. One ancilla qubit is the oracle's phase-kickback target.
# ---------------------------------------------------------------------------

def build_oracle(qc, q, ancilla):
    """
    Marks (phase-flips) exactly the basis states where q0 != q1 and q1 != q2,
    i.e. the proper 2-colorings of the path 0-1-2, using standard
    compute/mark/uncompute with two helper ("scratch") qubits that hold the
    XOR (not-equal) of each edge, then a Toffoli into the ancilla in the
    |-> state for phase kickback, then uncompute.
    """
    q0, q1, q2 = q[0], q[1], q[2]
    scratch = qc.qubits[3:5]  # two helper qubits: edge01 != , edge12 !=
    s01, s12 = scratch

    # compute s01 = q0 XOR q1, s12 = q1 XOR q2
    qc.cx(q0, s01)
    qc.cx(q1, s01)
    qc.cx(q1, s12)
    qc.cx(q2, s12)

    # mark when both s01 and s12 are 1 (both edges properly colored)
    qc.append(MCXGate(2), [s01, s12, ancilla])

    # uncompute
    qc.cx(q2, s12)
    qc.cx(q1, s12)
    qc.cx(q1, s01)
    qc.cx(q0, s01)


def build_diffuser(qc, q):
    qc.h(q)
    qc.x(q)
    # multi-controlled Z on all of q via H-MCX-H sandwich on the last qubit
    qc.h(q[-1])
    qc.append(MCXGate(len(q) - 1), list(q[:-1]) + [q[-1]])
    qc.h(q[-1])
    qc.x(q)
    qc.h(q)


def grover_p3_coloring_circuit(n_iterations):
    # qubits: 0,1,2 = vertex colors; 3,4 = oracle scratch; 5 = oracle ancilla
    qc = QuantumCircuit(6, 3)
    data = [0, 1, 2]

    qc.h(data)

    # ancilla in |-> for phase kickback
    qc.x(5)
    qc.h(5)

    for _ in range(n_iterations):
        build_oracle(qc, data, 5)
        build_diffuser(qc, data)

    qc.h(5)
    qc.x(5)

    qc.measure(data, [0, 1, 2])
    return qc


# Standard optimal Grover iteration count for M solutions out of N states.
N_SOLUTIONS_FOR_GROVER = N_SOLUTIONS
N_STATES = SEARCH_SPACE_SIZE
theta = math.asin(math.sqrt(N_SOLUTIONS_FOR_GROVER / N_STATES))
optimal_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = grover_p3_coloring_circuit(optimal_iterations)

sim = AerSimulator()
tqc = transpile(qc, sim)
job = sim.run(tqc, shots=4096)
result = job.result()
counts = result.get_counts()

print("\nGrover circuit: %d data qubits, %d iterations, %d shots"
      % (N_QUBITS, optimal_iterations, 4096))
print("Measurement counts (qiskit bit order, rightmost=q0):")
for bitstring, c in sorted(counts.items(), key=lambda kv: -kv[1]):
    print("  %s: %d" % (bitstring, c))

# Convert classical valid colorings (vertex0,vertex1,vertex2) into qiskit's
# little-endian measurement bitstring "c2 c1 c0" for the 3 measured (clbit
# 0,1,2 <- q0,q1,q2) results.
def coloring_to_bitstring(bits):
    v0, v1, v2 = bits
    return "%d%d%d" % (v2, v1, v0)

expected_bitstrings = set(coloring_to_bitstring(b) for b in VALID_COLORINGS)

total_shots = sum(counts.values())
solution_shots = sum(c for bs, c in counts.items() if bs in expected_bitstrings)
solution_fraction = solution_shots / total_shots

# top-2 most frequent measured outcomes should be exactly the 2 valid colorings
top2 = set(bs for bs, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:2])

print("\nExpected (classically valid) bitstrings: %s" % expected_bitstrings)
print("Top-2 most frequent measured bitstrings:  %s" % top2)
print("Fraction of shots landing on a valid coloring: %.3f" % solution_fraction)

verified = (top2 == expected_bitstrings) and (solution_fraction > 0.8)

if verified:
    print("\nPASS: Grover search on AerSimulator amplified exactly the "
          "classically-verified valid 2-colorings of P3 (chromatic number "
          "<= 2 witnessed by quantum search), matching the first-principles "
          "classical computation.")
else:
    print("\nFAIL: quantum search result did not match the classical "
          "ground truth.")
