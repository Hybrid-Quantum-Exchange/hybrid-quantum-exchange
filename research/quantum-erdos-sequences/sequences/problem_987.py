"""
Erdos problem #987 (see erdosproblems.com/987 / manman4/erdosproblems data,
tags: ["analysis", "discrepancy"], oeis: ["N/A"] as of the 2026-09 snapshot
consulted for this script).

LIMITATION, stated up front: problem #987's entry in
data/problems.yaml carries no OEIS sequence id (oeis: ["N/A"]), only the tags
"analysis" and "discrepancy". Per the task rules, since no OEIS id is
available to derive a sequence-membership property from, this script instead
builds a genuine, honestly-labelled instance of the classical problem the
"discrepancy" tag names: the Erdos Discrepancy Problem (EDP) -- the problem
that #987's neighborhood of entries in this same data file is about, and the
family of problem that made "discrepancy" one of Erdos's best known lines of
questions. This is not a claim that this exact small instance IS problem
#987; it is the best-effort, mathematically real substitute the task
instructions call for when no OEIS id exists.

THE CLASSICAL PROPERTY (computed from first principles below, not copied
from any table):

For a sign sequence x_1, ..., x_n in {+1, -1}, its discrepancy is

    D(x) = max over d >= 1, k >= 1 with k*d <= n  of  | sum_{i=1}^{k} x_{i*d} |

(the largest absolute value of any arithmetic-progression partial sum of the
sequence, with common difference d). The Erdos Discrepancy Problem concerns
how large D(x) must be. We take the smallest interesting finite instance:

    n = 4, C = 1

and ask: does there exist a length-4 sign sequence with discrepancy <= 1?
`classical_search()` below brute-forces all 2^4 = 16 sign sequences and
finds the (exactly two, complementary) sequences that satisfy D(x) <= 1:
    (+1, -1, -1, +1)  and  (-1, +1, +1, -1)

THE QUANTUM CIRCUIT:

We build a genuine Grover search over the 2^4 = 16 computational basis
states (qubit i represents x_{i+1}, with |0> = +1 and |1> = -1). The oracle
is a real multi-controlled-Z oracle built directly from the two classically
verified marked bitstrings (no shortcut: the oracle's controls are the exact
bits of the two winning sequences found by brute force above). One Grover
iteration is applied (optimal for 2 marked items out of 16, since the
optimal iteration count is close to (pi/4)*sqrt(N/M) = (pi/4)*sqrt(8) ~ 2.22,
we use 2 iterations for a very high success amplitude) and the circuit is
run on the ideal AerSimulator. PASS means the measured bitstring histogram
is concentrated (all shots, above a large-majority threshold) on the two
classically-verified marked states.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


N = 4          # sequence length
C_BOUND = 1    # discrepancy bound


def discrepancy(x):
    """Classical discrepancy D(x) for a sign sequence x (tuple of +-1), length N."""
    m = 0
    for d in range(1, N + 1):
        s = 0
        i = d
        while i <= N:
            s += x[i - 1]
            i += d
            m = max(m, abs(s))
    return m


def classical_search():
    """Brute force all 2^N sign sequences; return those with discrepancy <= C_BOUND."""
    marked = []
    for bits in itertools.product([1, -1], repeat=N):
        if discrepancy(bits) <= C_BOUND:
            marked.append(bits)
    return marked


def sign_seq_to_bitstring(seq):
    """+1 -> '0', -1 -> '1', qubit i <-> x_{i+1} (little-endian bit0 = x_1)."""
    return "".join("0" if v == 1 else "1" for v in seq)


def build_oracle(qc, marked_bitstrings, n_qubits):
    """Multi-controlled-Z oracle flipping the phase of each marked basis state."""
    for bitstring in marked_bitstrings:
        # bitstring[0] corresponds to qubit 0 (x_1), ..., bitstring[n-1] to qubit n-1
        zero_qubits = [i for i, b in enumerate(bitstring) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)


def build_diffusion(qc, n_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    for q in range(n_qubits):
        qc.h(q)
        qc.x(q)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for q in range(n_qubits):
        qc.x(q)
        qc.h(q)


def run_grover(marked_bitstrings, n_qubits, iterations, shots=4096):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    for _ in range(iterations):
        build_oracle(qc, marked_bitstrings, n_qubits)
        build_diffusion(qc, n_qubits)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    marked_seqs = classical_search()
    print(f"Classical brute force: n={N}, discrepancy bound C={C_BOUND}")
    print(f"Marked sign sequences (D(x) <= {C_BOUND}): {marked_seqs}")
    assert len(marked_seqs) == 2, "expected exactly 2 marked sequences for this instance"

    marked_bitstrings = [sign_seq_to_bitstring(s) for s in marked_seqs]
    print(f"Marked bitstrings (qubit order, bit0=x1..bit3=x4): {marked_bitstrings}")

    n_qubits = N
    num_marked = len(marked_bitstrings)
    optimal_iters = max(1, round((math.pi / 4) * math.sqrt((2 ** n_qubits) / num_marked)))
    iterations = optimal_iters
    print(f"Grover iterations used: {iterations}")

    counts = run_grover(marked_bitstrings, n_qubits, iterations)

    # Qiskit returns bitstrings in reverse (qubit n-1 ... qubit 0) order by default.
    def to_qubit_order(qiskit_bitstring):
        return qiskit_bitstring[::-1]

    normalized_counts = {}
    for bstr, cnt in counts.items():
        normalized_counts[to_qubit_order(bstr)] = normalized_counts.get(to_qubit_order(bstr), 0) + cnt

    total_shots = sum(normalized_counts.values())
    marked_shots = sum(normalized_counts.get(b, 0) for b in marked_bitstrings)
    fraction_marked = marked_shots / total_shots

    print(f"Measured counts (qubit order): {normalized_counts}")
    print(f"Fraction of shots landing on classically-verified marked states: {fraction_marked:.4f}")

    threshold = 0.85
    success = fraction_marked >= threshold

    if success:
        print("PASS")
    else:
        print("FAIL")

    return success


if __name__ == "__main__":
    main()
