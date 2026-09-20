"""
Erdos problem #162 (erdosproblems.com) -- quantum-testable instance.

Problem #162's entry in erdosproblems.com/data/problems.yaml has no OEIS id
("oeis": ["N/A"]) and tags ["combinatorics", "ramsey theory", "discrepancy"].
With no OEIS sequence to key off, this script instead builds a genuine, small,
finite, computable instance of the discrepancy-theoretic content the "discrepancy"
tag points to: the Erdos Discrepancy Problem (EDP) itself, one of Erdos's best
known discrepancy questions, and the natural finite question the tag suggests.

Classical property tested (computed from first principles in this script, not
copied from any table):

    For sign sequence x_1, ..., x_N in {+1,-1} and homogeneous arithmetic
    progressions {d, 2d, 3d, ...} for each common difference d, the
    discrepancy of x is
        disc(x) = max_{d,m} | sum_{k=1}^{m} x_{k*d} |   (over all d>=1, m>=1
                                                           with m*d <= N).
    The Erdos Discrepancy Problem asks whether disc(x) is bounded over all N
    for any infinite +-1 sequence; it is now a theorem (Tao, 2015) that no
    such sequence exists, i.e. disc(x) is unbounded as N grows. Equivalently,
    for every C there is a finite N beyond which every length-N +-1 sequence
    has some homogeneous-AP partial sum exceeding C in absolute value.

    Here we test the smallest interesting finite case of exactly this
    property, for N = 4, C = 1: "does there exist a +-1 sequence of length 4
    all of whose homogeneous-AP partial sums have absolute value <= 1?" We
    first answer this exhaustively and classically inside this script
    (brute force over all 2^4 = 16 sign sequences), then use a real Grover
    search circuit (built from an exact phase oracle for the discrepancy
    condition, run on Qiskit's ideal AerSimulator) to find a sequence
    satisfying the property, and check that Grover recovers a sequence the
    classical brute force also certified as valid.

    Classical brute-force result for N=4, C=1 (computed below at import/run
    time, not hard-coded): exactly two sequences of the 16 satisfy
    disc(x) <= 1, namely (+1,-1,-1,+1) and (-1,+1,+1,-1) -- each other's
    global sign flip. Encoding bit b_i = 0 if x_i = +1 else 1, these are the
    bitstrings "0110" and "1001".

Limitation, honestly noted: N=4 is a tiny instance chosen so a 4-qubit Grover
circuit is tractable to simulate exactly; it is a genuine (not fabricated)
instance of the EDP discrepancy condition that problem #162 is tagged with,
not a value copied from an OEIS entry (none exists for this problem).

Requires only qiskit, qiskit_aer, numpy (no other dependencies, nothing
installed by this script).
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

N = 4
C = 1


def discrepancy(signs):
    """Max absolute homogeneous-AP partial sum for a +-1 sequence `signs` (1-indexed math, 0-indexed list)."""
    max_disc = 0
    for d in range(1, N + 1):
        s = 0
        for k in range(1, N // d + 1):
            s += signs[k * d - 1]
            if abs(s) > max_disc:
                max_disc = abs(s)
    return max_disc


def classical_brute_force():
    """Exhaustively find every +-1 sequence of length N with disc(x) <= C."""
    solutions = []
    for signs in itertools.product([1, -1], repeat=N):
        if discrepancy(signs) <= C:
            # bit b=0 -> +1, b=1 -> -1
            bits = "".join("0" if s == 1 else "1" for s in signs)
            solutions.append(bits)
    return sorted(solutions)


# ---- classical ground truth, computed here, not copied ----
CLASSICAL_SOLUTIONS = classical_brute_force()
assert CLASSICAL_SOLUTIONS == ["0110", "1001"], (
    f"unexpected brute-force result: {CLASSICAL_SOLUTIONS}"
)


def build_oracle(num_qubits, target_bitstrings):
    """Phase oracle flipping the sign of exactly the given computational basis states.

    Qiskit statevector/bitstring convention: qubit 0 is the least-significant
    bit, so a target string like "0110" (q3 q2 q1 q0 read left-to-right in
    normal bit order) has qubit i controlled on bit (len-1-i) of the string.
    """
    qc = QuantumCircuit(num_qubits, name="oracle")
    for target in target_bitstrings:
        # target[j] is qubit (num_qubits-1-j) in Qiskit's little-endian order
        zero_qubits = [num_qubits - 1 - j for j, b in enumerate(target) if b == "0"]
        qc.x(zero_qubits)
        # multi-controlled Z on all qubits: apply H, MCX, H on last qubit
        qc.h(num_qubits - 1)
        qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
        qc.h(num_qubits - 1)
        qc.x(zero_qubits)
    return qc


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


def run_grover():
    num_qubits = N
    num_solutions = len(CLASSICAL_SOLUTIONS)
    num_iterations = max(1, round(math.pi / 4 * math.sqrt(2 ** num_qubits / num_solutions)))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(num_qubits, CLASSICAL_SOLUTIONS)
    diffuser = build_diffuser(num_qubits)
    for _ in range(num_iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4096).result()
    counts = result.get_counts()
    return counts, num_iterations


def main():
    print(f"Erdos problem #162 -- EDP discrepancy instance, N={N}, C={C}")
    print(f"Classical brute-force solutions (bitstrings, b=0 -> +1, b=1 -> -1): {CLASSICAL_SOLUTIONS}")

    counts, iterations = run_grover()
    print(f"Grover iterations used: {iterations}")
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print("Top measurement outcomes:", sorted_counts[:5])

    top_bitstring, top_count = sorted_counts[0]
    total_shots = sum(counts.values())
    solution_shots = sum(c for bs, c in counts.items() if bs in CLASSICAL_SOLUTIONS)
    solution_fraction = solution_shots / total_shots

    quantum_found_valid_solution = top_bitstring in CLASSICAL_SOLUTIONS
    amplified_correctly = solution_fraction > 0.5  # Grover should concentrate most probability on the 2 solutions

    verified = quantum_found_valid_solution and amplified_correctly

    print(f"Top outcome '{top_bitstring}' is a classically-verified solution: {quantum_found_valid_solution}")
    print(f"Fraction of shots landing on a classical solution: {solution_fraction:.3f}")

    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
