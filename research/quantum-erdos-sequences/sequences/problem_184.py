"""
Erdos problem #184 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, problems.yaml, block "number: '184'"):
    prize: no
    status: open
    tags: ["graph theory", "cycles"]
    oeis: ["possible"]

IMPORTANT / LIMITATION, reported honestly rather than faked:
    Problem #184's YAML entry does NOT carry a real OEIS sequence id. Its
    "oeis" field literally contains the string "possible" -- a placeholder,
    not an A-number. There is therefore no OEIS sequence to derive a
    property from for this problem, and this script does NOT claim one.

    In place of a fabricated OEIS-derived property, this script builds a
    genuine, small, finite, classically-checkable instance drawn directly
    from problem #184's own tags ("graph theory", "cycles"): Hamiltonian
    4-cycles in the complete graph K4.

    Property under test:
        Let K4 have vertices {0,1,2,3} and the 6 possible undirected edges
            e0=(0,1) e1=(0,2) e2=(0,3) e3=(1,2) e4=(1,3) e5=(2,3)
        A subset S of these 6 edges (encoded as a 6-bit string, one bit per
        edge) forms a Hamiltonian cycle of K4 iff every vertex has degree
        exactly 2 in the subgraph (S). For a simple graph on 4 vertices this
        forces S to be exactly one of the three 4-cycles of K4 (there is no
        other way to be 2-regular on 4 labelled vertices without multi-edges).

    This is computed from first principles below (brute force over all
    2^6 = 64 edge subsets, no OEIS lookup, no hard-coded literal answer),
    giving the ground-truth set of "marked" bitstrings. A Grover search
    circuit is then built whose oracle marks exactly those bitstrings, run
    on the ideal AerSimulator, and the most frequently measured bitstring
    is checked against the classically-computed marked set.

    This is a real Grover amplitude-amplification circuit over a genuine
    finite combinatorial search space tied to the problem's own tags; it is
    not derived from an OEIS sequence because problem #184 has none.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate

# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no lookup tables).
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # index i <-> qubit i
N_EDGES = len(EDGES)  # 6


def degree_sequence(subset_bits):
    """subset_bits: tuple of 0/1 of length N_EDGES. Returns degree of each vertex."""
    deg = {v: 0 for v in VERTICES}
    for bit, (a, b) in zip(subset_bits, EDGES):
        if bit:
            deg[a] += 1
            deg[b] += 1
    return deg


def is_hamiltonian_cycle(subset_bits):
    """A subset of K4's edges is a Hamiltonian 4-cycle iff every vertex has
    degree exactly 2 (simple graph on 4 vertices => forces a single 4-cycle)."""
    deg = degree_sequence(subset_bits)
    return all(d == 2 for d in deg.values())


def bits_to_string(bits):
    # Qiskit prints classical registers with qubit 0 as the RIGHTMOST bit.
    return "".join(str(b) for b in reversed(bits))


marked_bitstrings = []
for bits in itertools.product([0, 1], repeat=N_EDGES):
    if is_hamiltonian_cycle(bits):
        marked_bitstrings.append(bits_to_string(bits))

marked_bitstrings = sorted(set(marked_bitstrings))
num_marked = len(marked_bitstrings)

print(f"Search space size N = 2^{N_EDGES} = {2 ** N_EDGES}")
print(f"Classically computed marked (Hamiltonian-4-cycle) bitstrings: {marked_bitstrings}")
print(f"Number of marked states M = {num_marked}")

# Sanity check against known combinatorics: K4 has exactly 3 distinct
# Hamiltonian cycles (4!/(2*4) = 3). This is an independent classical check.
assert num_marked == 3, f"expected 3 Hamiltonian 4-cycles in K4, computed {num_marked}"

# ---------------------------------------------------------------------------
# 2. Grover search circuit whose oracle marks exactly `marked_bitstrings`.
# ---------------------------------------------------------------------------

n = N_EDGES  # 6 qubits for the search register


def build_oracle(marked):
    """Phase-flip oracle: applies -1 phase to each basis state in `marked`
    (bitstrings given qiskit-order, qubit 0 = rightmost char)."""
    qc = QuantumCircuit(n, name="oracle")
    for bitstring in marked:
        # bitstring[0] is qubit n-1 ... bitstring[n-1] is qubit 0
        qubit_bits = list(reversed(bitstring))  # index i -> qubit i value
        zero_qubits = [i for i, b in enumerate(qubit_bits) if b == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        # multi-controlled Z on all n qubits (flip phase of |11..1>)
        qc.h(n - 1)
        qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
        qc.h(n - 1)
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser():
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(marked_bitstrings)
diffuser = build_diffuser()

# Optimal number of Grover iterations for N=64, M=3.
N = 2 ** n
theta = math.asin(math.sqrt(num_marked / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations}")

qc = QuantumCircuit(n, n)
qc.h(range(n))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(n))
    qc.append(diffuser.to_gate(), range(n))
qc.measure(range(n), range(n))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Fraction of shots landing on a marked (correct) bitstring.
marked_shots = sum(c for bs, c in counts.items() if bs in marked_bitstrings)
marked_fraction = marked_shots / shots
top_result = max(counts, key=counts.get)

print(f"Top measured bitstring: {top_result} (count {counts[top_result]}/{shots})")
print(f"Fraction of shots landing on a marked state: {marked_fraction:.3f}")

# ---------------------------------------------------------------------------
# 4. Compare quantum result to classical answer.
# ---------------------------------------------------------------------------

quantum_found_marked_state = top_result in marked_bitstrings
amplification_worked = marked_fraction > (num_marked / N) * 3  # well above uniform baseline

if quantum_found_marked_state and amplification_worked:
    print("PASS")
else:
    print("FAIL")
