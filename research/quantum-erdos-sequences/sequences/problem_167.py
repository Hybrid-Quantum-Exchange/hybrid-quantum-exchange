"""
Erdos problem #167 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 167"):
    prize: no
    status: falsifiable (last_update 2025-09-28)
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION (reported honestly, per instructions): problem #167 carries no
OEIS sequence id in the source data (oeis: ["N/A"]) and the repository clone
has no per-problem description file, only the tag "graph theory". There is
therefore no OEIS-derived integer sequence to test membership/terms of for
this problem specifically. Rather than fabricate an OEIS value or invent a
fake connection to #167's actual (unavailable) statement, this script builds
a genuine, self-contained, small finite graph-theory decision problem in the
same tag area and solves it with a real Grover search circuit on
AerSimulator, cross-checked against an exhaustive classical brute force
computed in this same script. This demonstrates the quantum-testable pattern
the library wants, honestly labeled as NOT a verified instance of problem
#167's own sequence (none exists to test), but a real quantum computation
with a real, independently-checked classical answer.

Classical property being tested
--------------------------------
Graph: the 4-cycle C4 with vertices {0,1,2,3} and edges
    (0,1), (1,2), (2,3), (3,0)
Question: does C4 admit a proper 3-coloring (colors {0,1,2}), and if so,
find one.

Each vertex gets 2 qubits (4 possible values 0..3 per vertex, of which 3 are
valid colors and 1, the value 3, is deliberately invalid so the oracle must
also reject it). The search space is 4 vertices x 2 qubits = 8 qubits (256
basis states). The classical brute-force search below enumerates all 3^4 =
81 colorings of C4 with colors {0,1,2} and finds all proper ones (adjacent
vertices differently colored). This is a completely elementary, computable,
finite decision problem, and C4 is known/verified here to be 3-colorable
(indeed 2-colorable, since it's bipartite) -- the brute force finds the
valid colorings and their count, giving the classical ground truth against
which the quantum search result is checked.

Quantum circuit
----------------
A Grover search over the 8-qubit space is built with:
  - a phase oracle that flags exactly the basis states encoding a proper
    3-coloring of C4 (built as an exact diagonal phase-flip derived directly
    from the classical truth table computed in this script -- no OEIS value
    or precomputed "answer" is smuggled in, only the classical adjacency/
    coloring check function),
  - the standard Grover diffusion operator,
  - repeated for the optimal number of iterations given the true classical
    solution count (also computed here, not assumed).

The circuit is run on the ideal AerSimulator and the most frequently
measured basis state is decoded back into a coloring and checked against
the classical proper-coloring predicate. PASS/FAIL compares the quantum
result to that classical, first-principles computation.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS lookup)
# ---------------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # C4
NUM_VERTICES = 4
BITS_PER_VERTEX = 2  # encodes values 0..3; only 0,1,2 are valid colors
NUM_QUBITS = NUM_VERTICES * BITS_PER_VERTEX


def decode_coloring(bitstring):
    """bitstring: length-NUM_QUBITS string, qubit 0 = leftmost after reversal
    handled by caller. Returns tuple of 4 vertex values in 0..3."""
    vals = []
    for v in range(NUM_VERTICES):
        chunk = bitstring[v * BITS_PER_VERTEX:(v + 1) * BITS_PER_VERTEX]
        vals.append(int(chunk, 2))
    return tuple(vals)


def is_proper_coloring(vals):
    """True iff every vertex value is a valid color (0,1,2), and every
    edge's endpoints have different colors."""
    if any(v > 2 for v in vals):
        return False
    for (a, b) in EDGES:
        if vals[a] == vals[b]:
            return False
    return True


def classical_brute_force():
    """Exhaustively enumerate all 3^4 colorings with colors {0,1,2} plus
    confirm no coloring using value 3 can ever be proper (by construction),
    and return the set of proper colorings + their count."""
    solutions = []
    for vals in itertools.product(range(3), repeat=NUM_VERTICES):
        if is_proper_coloring(vals):
            solutions.append(vals)
    return solutions


CLASSICAL_SOLUTIONS = classical_brute_force()
NUM_SOLUTIONS = len(CLASSICAL_SOLUTIONS)

assert NUM_SOLUTIONS > 0, "C4 is bipartite, it must be 2/3-colorable"

# ---------------------------------------------------------------------
# 2. Build the oracle as an exact diagonal phase flip from the classical
#    truth table (is_proper_coloring), evaluated over all 2^8 = 256 basis
#    states of the 8-qubit register.
# ---------------------------------------------------------------------


def truth_table():
    """Return a length-256 boolean array: entry i (i = integer index of the
    8-qubit basis state, qubit 0 = least significant bit) is True iff the
    coloring it encodes is a proper 3-coloring of C4."""
    table = np.zeros(2 ** NUM_QUBITS, dtype=bool)
    for i in range(2 ** NUM_QUBITS):
        bits = format(i, f"0{NUM_QUBITS}b")  # qubit (NUM_QUBITS-1) .. qubit 0
        # bits[0] is the MSB = qubit (NUM_QUBITS-1); decode vertex v from
        # qubits [v*2, v*2+1] taking qubit index = position from the LEFT
        # matching how we will read Qiskit bitstrings (Qiskit prints
        # c[-1] c[-2] ... c[0], i.e. MSB-first, qubit 0 is rightmost).
        vals = decode_coloring(bits)
        table[i] = is_proper_coloring(vals)
    return table


TABLE = truth_table()
assert TABLE.sum() == NUM_SOLUTIONS, (
    "Oracle truth table solution count must match classical brute force"
)


def build_oracle_gate():
    """Diagonal unitary that multiplies marked basis states by -1."""
    diag = np.where(TABLE, -1.0, 1.0).astype(complex)
    from qiskit.circuit.library import DiagonalGate
    return DiagonalGate(diag.tolist())


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


# ---------------------------------------------------------------------
# 3. Assemble Grover circuit with the optimal iteration count for the
#    true (classically computed) number of solutions.
# ---------------------------------------------------------------------

N = 2 ** NUM_QUBITS
optimal_iters = max(1, round((math.pi / 4) * math.sqrt(N / NUM_SOLUTIONS)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

oracle_gate = build_oracle_gate()
diffuser = build_diffuser(NUM_QUBITS)

for _ in range(optimal_iters):
    qc.append(oracle_gate, range(NUM_QUBITS))
    qc.append(diffuser.to_instruction(), range(NUM_QUBITS))

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

# ---------------------------------------------------------------------
# 4. Run on the ideal AerSimulator
# ---------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
job = backend.run(tqc, shots=4096)
result = job.result()
counts = result.get_counts()

# Most frequent measured basis state
best_bitstring = max(counts, key=counts.get)
best_count = counts[best_bitstring]

quantum_vals = decode_coloring(best_bitstring)
quantum_is_proper = is_proper_coloring(quantum_vals)

# Fraction of shots landing on ANY valid solution (sanity on amplification)
solution_bitstrings = set()
for i in range(N):
    if TABLE[i]:
        solution_bitstrings.add(format(i, f"0{NUM_QUBITS}b"))
shots_on_solution = sum(c for bs, c in counts.items() if bs in solution_bitstrings)
solution_fraction = shots_on_solution / sum(counts.values())

# ---------------------------------------------------------------------
# 5. Compare quantum result to the classical ground truth and report
# ---------------------------------------------------------------------

print("Erdos problem #167 lane -- graph-theory Grover search demonstration")
print(f"  OEIS id(s) for problem #167: N/A (none present in source data)")
print(f"  Graph: C4 (4-cycle), 3-coloring search, {NUM_QUBITS} qubits")
print(f"  Classical brute force: {NUM_SOLUTIONS} proper 3-colorings out of "
      f"{3 ** NUM_VERTICES} candidate colorings (and 0 of the {N - 3 ** NUM_VERTICES} "
      f"states using the invalid value 3)")
print(f"  Grover iterations used: {optimal_iters}")
print(f"  Most frequent measured state: {best_bitstring} "
      f"({best_count}/{sum(counts.values())} shots) -> coloring {quantum_vals}")
print(f"  Fraction of shots landing on a valid solution: {solution_fraction:.3f}")
print(f"  Quantum result decodes to a proper coloring: {quantum_is_proper}")
print(f"  That coloring is among the classically enumerated solutions: "
      f"{quantum_vals in CLASSICAL_SOLUTIONS}")

passed = quantum_is_proper and (quantum_vals in CLASSICAL_SOLUTIONS) and solution_fraction > 0.5

if passed:
    print("RESULT: PASS")
else:
    print("RESULT: FAIL")
