"""
Erdos problem #171 (erdosproblems.com) -- quantum-testable instance.

Problem #171 is the density Hales-Jewett problem; its metadata in
erdosproblems' data/problems.yaml lists OEIS id A156989 ("Density
Hales-Jewett numbers") and tags ["additive combinatorics", "combinatorics"].
The true DHJ numbers c(n, 3) (smallest N such that every 2-coloring of
{1,...,N}^n has a monochromatic combinatorial line, alphabet size 3) grow
astronomically (already c(1,3)=3, c(2,3)=? is open/huge by known bounds), so
no interesting term of A156989 itself fits on a handful of qubits. Rather
than fabricate a fake "N is in A156989" oracle, this script tests the exact
combinatorial object the sequence counts thresholds for: the existence of a
monochromatic combinatorial line in a 2-coloring of the grid {0,1,2}^2
(n = 2 dimensions, alphabet size 3 -- the smallest non-trivial DHJ instance).

Classical property tested
--------------------------
Fix the grid {0,1,2}^2 (9 points) and the explicit 2-coloring
    color(i, j) = (i + 2*j) mod 2.
A "combinatorial line" is obtained by choosing a non-empty subset of the two
coordinates to be a wildcard x and sweeping x over {0,1,2} while the other
coordinate(s) stay fixed; there are exactly (3+1)^2 - 3^2 = 7 such lines for
n = 2, alphabet 3. The script enumerates all 7 lines by exhaustive Python
(first-principles ground truth) and evaluates, for each, whether all 3 of its
points get the same color. This is exactly the finite object A156989's
defining threshold (density-Hales-Jewett numbers) is about: whether a
coloring of a grid over an alphabet forces a monochromatic line.

For the chosen coloring the classical (first-principles) computation below
finds exactly 3 monochromatic lines, at line-indices {0, 1, 2} out of the 7
(the three "row" lines i = 0, 1, 2 with j wildcard).

Quantum circuit
----------------
We run Grover's algorithm over a 3-qubit index register (8 basis states,
indices 0..6 valid lines, index 7 unused/never marked) whose oracle marks
exactly the indices found monochromatic classically ({0, 1, 2}), built with
standard multi-controlled-Z "mark this bitstring" gates (one per marked
index) sandwiched between X gates that flip 0-bits to 1-bits. After the
optimal number of Grover iterations for N = 8, M = 3
(floor(pi/4 * sqrt(N/M)) = 1 iteration), we measure and take the most
frequent outcomes. If Grover amplifies exactly the classically-marked set of
monochromatic-line indices, the quantum result is compared against the
classical answer and PASS/FAIL is printed accordingly.
"""

import itertools
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS values copied)
# ---------------------------------------------------------------------------

ALPHABET = [0, 1, 2]


def color(i, j):
    return (i + 2 * j) % 2


def enumerate_lines():
    """All combinatorial lines of {0,1,2}^2: each coordinate slot is a fixed
    value in {0,1,2} or a wildcard 'x', at least one wildcard."""
    lines = []
    for a in ALPHABET + ["x"]:
        for b in ALPHABET + ["x"]:
            if a == "x" or b == "x":
                lines.append((a, b))
    return lines


def line_points(line):
    a, b = line
    pts = []
    for v in ALPHABET:
        i = v if a == "x" else a
        j = v if b == "x" else b
        pts.append((i, j))
    return pts


def classical_monochromatic_line_indices():
    lines = enumerate_lines()
    marked = []
    for idx, line in enumerate(lines):
        cols = [color(i, j) for (i, j) in line_points(line)]
        if len(set(cols)) == 1:
            marked.append(idx)
    return lines, marked


LINES, MARKED_INDICES = classical_monochromatic_line_indices()
assert len(LINES) == 7, "expected exactly 7 combinatorial lines for n=2, alphabet 3"

N_QUBITS = 3  # ceil(log2(7)) = 3, index 7 is unused padding (never marked)
N_STATES = 2 ** N_QUBITS


def index_to_bits(idx, n):
    return [(idx >> k) & 1 for k in range(n)][::-1]  # MSB-first list of 0/1


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search marking the classically-monochromatic
#    line indices
# ---------------------------------------------------------------------------

def mark_index_gate(qc, idx, n_qubits):
    """Apply a phase flip (-1) to basis state |idx> via X-sandwiched MCZ."""
    bits = index_to_bits(idx, n_qubits)
    flip_qubits = [n_qubits - 1 - pos for pos, b in enumerate(bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for q in flip_qubits:
        qc.x(q)


def build_oracle(n_qubits, marked_indices):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        mark_index_gate(qc, idx, n_qubits)
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


def grover_iterations(n_states, n_marked):
    import math
    if n_marked == 0:
        return 0
    ratio = n_states / n_marked
    return max(1, round((math.pi / 4) * math.sqrt(ratio)))


def build_grover_circuit(n_qubits, marked_indices):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_indices)
    diffuser = build_diffuser(n_qubits)

    iterations = grover_iterations(2 ** n_qubits, len(marked_indices))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


# ---------------------------------------------------------------------------
# 3. Run and compare
# ---------------------------------------------------------------------------

def main():
    print("Erdos problem #171 -- density Hales-Jewett (OEIS A156989)")
    print(f"Grid {{0,1,2}}^2, coloring color(i,j) = (i + 2j) mod 2")
    print(f"All {len(LINES)} combinatorial lines: {LINES}")
    print(f"Classically monochromatic line indices: {MARKED_INDICES}")

    qc, iterations = build_grover_circuit(N_QUBITS, MARKED_INDICES)
    print(f"Grover iterations used: {iterations} (N={N_STATES}, M={len(MARKED_INDICES)})")

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit order: rightmost char = qubit 0. We built index bits MSB-first
    # onto qubits [n-1 .. 0], i.e. qubit (n-1-pos) holds bit `pos` (MSB-first).
    # The classical index recoverable from a bitstring 'c2 c1 c0' (as printed,
    # MSB..LSB of qubit index) is simply int(bitstring, 2) after reversing to
    # match our MSB-first convention on qubits high->low.
    def bitstring_to_index(bs):
        # qiskit returns c_{n-1}...c_0 (c0 = qubit0, leftmost is highest qubit)
        return int(bs, 2)

    index_counts = {}
    for bitstring, cnt in counts.items():
        idx = bitstring_to_index(bitstring)
        index_counts[idx] = index_counts.get(idx, 0) + cnt

    sorted_by_count = sorted(index_counts.items(), key=lambda kv: -kv[1])
    print(f"Measured index counts (top): {sorted_by_count[:7]}")

    top_k = len(MARKED_INDICES)
    quantum_top_indices = set(idx for idx, _ in sorted_by_count[:top_k])
    classical_marked = set(MARKED_INDICES)

    marked_prob_mass = sum(index_counts.get(i, 0) for i in classical_marked) / shots
    unmarked_prob_mass = 1.0 - marked_prob_mass

    print(f"Classical marked set: {classical_marked}")
    print(f"Quantum top-{top_k} measured set: {quantum_top_indices}")
    print(f"Probability mass on classically-marked indices: {marked_prob_mass:.3f}")

    verified = (quantum_top_indices == classical_marked) and (marked_prob_mass > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
