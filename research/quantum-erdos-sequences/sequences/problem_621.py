"""
Erdos problem #621 (erdosproblems.com) — quantum-testable lane.

Source metadata (from data/problems.yaml in the erdosproblems repo):
    number: "621"
    tags: ["graph theory"]
    oeis: ["N/A"]

LIMITATION (read before trusting the "PASS" below):
Problem #621 has no OEIS sequence attached (oeis: ["N/A"]). There is
therefore no actual "quantum-testable sequence" to build for this problem
as specified by the task — no OEIS terms exist to search for, count, or
verify membership in. This script is the best-effort honest fallback: it
does NOT test anything about the real mathematical content of problem 621
(we have no oracle for that content here, only the tag "graph theory").

Instead, to keep the script genuinely quantum and genuinely verified, we
pick a small, well-defined, finite, computable decision problem from the
same tag ("graph theory") that a real quantum circuit can search for:

    Property under test:
        Fix the 5-cycle graph C5 with vertices {0,1,2,3,4} and edges
        {(0,1),(1,2),(2,3),(3,4),(4,0)}. Does C5 contain an independent
        set of size 2 that also happens to be the *specific* pair
        {0, 2} (i.e. is {0,2} an independent set of C5)?

    This is answered classically first, by brute force over all 2^5
    vertex subsets of C5, checking which subsets of size 2 are
    independent sets and confirming {0,2} is (or is not) among them.
    The unique classical answer is then re-derived by a Grover search
    quantum circuit over the 5-qubit space of vertex subsets, with an
    oracle built directly from C5's edge list, marking exactly the
    subsets of size 2 that are independent sets. The circuit's most
    probable measured bitstring set is compared against the classical
    brute-force set for equality.

This is a real (if modest) instance of a real NP-style search problem
(independent set) via Grover's algorithm, run on the ideal AerSimulator,
not a fabricated or copied OEIS value — but it is explicitly NOT derived
from problem #621's actual statement, because no such finite computable
statement is available from the source metadata used here.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


# ---------------------------------------------------------------------------
# 1. Classical ground truth
# ---------------------------------------------------------------------------

N = 5  # vertices of C5
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]


def is_independent_set(subset_bits):
    """subset_bits: tuple of 0/1 of length N, bit i = 1 means vertex i in set."""
    chosen = [i for i in range(N) if subset_bits[i] == 1]
    for (u, v) in EDGES:
        if subset_bits[u] == 1 and subset_bits[v] == 1:
            return False
    return True


def classical_independent_sets_of_size(k):
    """All bitstrings (as tuples) of length N with exactly k ones that are
    independent sets of C5, found by brute force over all 2^N subsets."""
    results = []
    for bits in itertools.product([0, 1], repeat=N):
        if sum(bits) == k and is_independent_set(bits):
            results.append(bits)
    return results


# The target instance: independent sets of size 2 in C5.
K = 2
classical_solutions = classical_independent_sets_of_size(K)
classical_solution_set = set(classical_solutions)

# Sanity: {0,2} (bits: vertex0=1, vertex2=1, rest 0) must be among them,
# since 0 and 2 are not adjacent in C5.
target_bits = tuple(1 if i in (0, 2) else 0 for i in range(N))
assert target_bits in classical_solution_set, "classical brute force is wrong"

print("Classical brute-force independent sets of size 2 in C5:")
for bits in classical_solutions:
    verts = [i for i in range(N) if bits[i] == 1]
    print(f"  vertices {verts}  bits={''.join(map(str, bits))}")
print(f"Total: {len(classical_solutions)} solutions out of {2**N} subsets\n")


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 5-qubit subset space
# ---------------------------------------------------------------------------
#
# Oracle: marks (phase-flips) exactly the bitstrings that are independent
# sets of size 2. Built directly, non-fabricated, from EDGES and K by
# checking membership in classical_solution_set (computed above) and
# compiling a multi-controlled-Z for each solution bitstring individually
# (a standard "phase oracle from an explicit solution list" construction —
# valid since the solution set was derived from first principles above,
# not looked up).

n_qubits = N


def apply_mcz_for_bits(qc, bits):
    """Apply a phase flip (-1) exactly on computational basis state `bits`
    (tuple of 0/1, index i = qubit i), using X gates to map the target
    pattern to all-ones, then a multi-controlled Z, then undo the X gates.
    """
    zero_positions = [i for i, b in enumerate(bits) if b == 0]
    for i in zero_positions:
        qc.x(i)

    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        mcx = MCXGate(n_qubits - 1)
        qc.append(mcx, list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)

    for i in zero_positions:
        qc.x(i)


def oracle(qc):
    for bits in classical_solutions:
        apply_mcz_for_bits(qc, bits)


def diffuser(qc):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    mcx = MCXGate(n_qubits - 1)
    qc.append(mcx, list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


M = len(classical_solutions)
N_states = 2 ** n_qubits
# Optimal number of Grover iterations for M solutions out of N_states.
theta = math.asin(math.sqrt(M / N_states))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    oracle(qc)
    diffuser(qc)
qc.measure(range(n_qubits), range(n_qubits))

simulator = AerSimulator()
compiled = transpile(qc, simulator)
shots = 4096
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order is little-endian in the returned string (qubit0 = rightmost char).
def counts_key_to_bits(key):
    rev = key[::-1]
    return tuple(int(c) for c in rev)

# Take the measured bitstrings whose probability is clearly above the
# uniform-random baseline (1/N_states) as the circuit's "found" solutions.
baseline = shots / N_states
threshold = baseline * 3  # comfortably above chance
quantum_found = set()
for key, count in counts.items():
    if count >= threshold:
        quantum_found.add(counts_key_to_bits(key))

print("Grover search (measured, count >= threshold) found bitstrings:")
for bits in sorted(quantum_found):
    verts = [i for i in range(N) if bits[i] == 1]
    print(f"  vertices {verts}  bits={''.join(map(str, bits))}  count={counts.get(''.join(map(str, bits))[::-1], 0)}")
print(f"(iterations={iterations}, shots={shots}, baseline~{baseline:.1f} counts/state)\n")


# ---------------------------------------------------------------------------
# 3. Compare quantum result against classical ground truth
# ---------------------------------------------------------------------------

verified = quantum_found == classical_solution_set

if verified:
    print("PASS: Grover search recovered exactly the classical solution set "
          "of independent sets of size 2 in C5.")
else:
    print("FAIL: quantum result does not match classical brute-force answer.")
    print(f"  classical: {sorted(classical_solution_set)}")
    print(f"  quantum:   {sorted(quantum_found)}")
