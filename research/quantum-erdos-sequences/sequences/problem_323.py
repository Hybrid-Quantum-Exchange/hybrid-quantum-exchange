"""
Erdos problem #323 -- quantum-testable instance.

Erdos problem 323 (informal, open) concerns numbers representable as sums of
a bounded number of positive k-th powers (a Waring-type question). The
problems.yaml entry for #323 lists OEIS ids A004825, A004831, A004832,
A004833, A004842, A004843, A004844, A004845, A004857, A004869 -- each of the
form "numbers that are the sum of at most m positive k-th powers" for a
range of (k, m) pairs (confirmed directly against oeis.org, e.g. A004831:
"Numbers that are the sum of at most 2 nonzero 4th powers.").

This script uses A004831: numbers that are the sum of at most 2 nonzero
4th powers.

Classical property tested
--------------------------
Fix N = 17 and search space a, b in {1, 2, 3} (2 qubits each, so 4th
powers 1, 16, 81 are reachable; a and b range over nonzero values only,
matching "positive" in the OEIS definition). The property is:

    Is there a pair (a, b) in {1,2,3}^2 with a^4 + b^4 == 17?

The classical answer, computed by brute force in this script before any
quantum code runs, is that (a, b) = (1, 2) and (2, 1) are the only two
solutions in the search space (1^4 + 2^4 = 1 + 16 = 17), so 17 is indeed a
member of A004831, and the marked set for Grover search has exactly 2 out
of 16 basis states.

Quantum approach
-----------------
Grover's algorithm on 4 qubits (2 qubits for a-1, 2 qubits for b-1, each
encoding values 0..3 i.e. a,b in 1..4, restricted in the oracle to a,b<=3
consistent with the classical search space above). The oracle is built as
an exact diagonal phase-flip unitary derived directly from the classical
brute-force solution set (not a hand-waved guess), then the standard
Grover diffusion operator is applied for the optimal number of iterations
for a 16-item space with 2 marked items. The circuit is run on the ideal
AerSimulator, and the most frequently measured basis states are compared
against the classical solution set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator


def classical_solutions(N, max_val=3):
    """Brute-force all (a, b) in [1, max_val]^2 with a^4 + b^4 == N."""
    sols = []
    for a in range(1, max_val + 1):
        for b in range(1, max_val + 1):
            if a ** 4 + b ** 4 == N:
                sols.append((a, b))
    return sols


def index_of(a, b, max_val=3):
    """Map (a, b) in [1, max_val]^2 to a 4-qubit computational basis index.

    2 qubits encode (a-1) in [0,3], 2 qubits encode (b-1) in [0,3];
    values with a>max_val or b>max_val never arise here since a,b are
    drawn from [1, max_val].
    """
    return (b - 1) * 4 + (a - 1)


def build_oracle(marked_indices, n_qubits):
    """Exact diagonal phase-flip oracle: -1 on marked indices, +1 elsewhere."""
    diag = [1.0] * (2 ** n_qubits)
    for idx in marked_indices:
        diag[idx] = -1.0
    return DiagonalGate(diag)


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits - 1 >= 1:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    else:
        qc.z(0)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    N = 17
    MAX_VAL = 3
    n_qubits = 4  # 2 for a-1, 2 for b-1, values 0..3 -> a,b in 1..4

    # --- classical ground truth, computed here from first principles ---
    sols = classical_solutions(N, MAX_VAL)
    assert sols == [(1, 2), (2, 1)], f"unexpected classical solutions: {sols}"
    marked_indices = sorted(index_of(a, b, MAX_VAL) for (a, b) in sols)
    classical_member = len(sols) > 0  # is 17 a sum of at most 2 nonzero 4th powers?

    print(f"N = {N}")
    print(f"Classical brute-force solutions (a,b) with a^4+b^4=N: {sols}")
    print(f"17 in A004831 (sum of <=2 nonzero 4th powers)? {classical_member}")
    print(f"Marked basis-state indices for Grover oracle: {marked_indices}")

    # --- Grover search over the 16-state space for those indices ---
    N_states = 2 ** n_qubits
    M = len(marked_indices)
    theta = math.asin(math.sqrt(M / N_states))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked_indices, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle, range(n_qubits))
        qc.compose(diffuser, range(n_qubits), inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit ordering: rightmost classical bit = qubit 0.
    def bitstring_to_index(bs):
        # Qiskit's count keys are "c_{n-1}...c_1 c_0"; c_0 (rightmost char)
        # was measured from qubit 0, which is the least-significant bit of
        # our index encoding, so a plain binary parse already lines up.
        return int(bs, 2)

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_indices = set()
    total = sum(counts.values())
    running = 0
    for bs, cnt in sorted_counts:
        idx = bitstring_to_index(bs)
        top_indices.add(idx)
        running += cnt
        if len(top_indices) >= M and running / total > 0.7:
            break

    print(f"Grover run: {iterations} iteration(s), {shots} shots")
    print(f"Top measured indices: {sorted(top_indices)} (expected: {marked_indices})")

    quantum_found_correct_set = set(marked_indices).issubset(top_indices)

    passed = classical_member and quantum_found_correct_set
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    import sys
    ok = main()
    sys.exit(0 if ok else 1)
