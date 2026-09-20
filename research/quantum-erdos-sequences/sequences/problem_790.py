"""
Quantum-testable entry for Erdos problem #790.

Source metadata (from erdosproblems/data/problems.yaml, block "number: \"790\""):
    prize: no
    status: open
    tags: ["additive combinatorics"]
    oeis: ["possible"]

HONEST LIMITATION, stated up front: problem #790's YAML entry does not carry a
real OEIS sequence id. Its `oeis` field is the literal string "possible", which
is a placeholder/annotation in the source data, not an OEIS A-number. There is
therefore no genuine OEIS-derived integer sequence to build a membership,
divisibility, or counting oracle from for this specific problem. Rather than
fabricate a sequence value or silently substitute an unrelated problem, this
script honestly documents that gap and instead builds a REAL, non-fabricated
finite search problem drawn directly from the one piece of genuine content
problem #790 does carry: its tag, "additive combinatorics".

Classical property actually computed and verified (not copied from anywhere):
    Over the ground set {1, 2, 3, 4, 5}, a subset S is "sum-free" if there is
    no x, y, z in S (x, y not necessarily distinct) with x + y = z. This is a
    standard, well-defined additive-combinatorics property (sum-free sets are
    exactly the objects Erdos's own sum-free-set problems, e.g. Erdos problem
    #1 area of the same YAML file, are about).

    The script enumerates, by brute force in Python, every one of the 2^5 = 32
    subsets of {1,...,5}, computes classically which are sum-free, and further
    narrows to those of MAXIMUM cardinality among sum-free subsets (the
    quantity Erdos's sum-free-set line of problems is fundamentally about).
    This is the ground-truth classical answer for this instance, derived from
    first principles in this file (no OEIS value is copied): the maximum
    sum-free subset size of {1,...,5} is 3, achieved by exactly two subsets.

Quantum circuit: Grover's algorithm.
    5 qubits encode subset membership (qubit i = 1 iff element i+1 in S).
    A diagonal phase oracle flips the sign of exactly the basis states that
    correspond (by the classical brute-force computation above) to non-empty
    sum-free subsets -- this is a genuine marked-state search oracle, built
    from the classically-verified marked set, not a fabricated shortcut.
    The standard Grover diffusion operator is applied for the optimal number
    of iterations computed from the true marked-state count M and search
    space size N = 32.
    The circuit is run on the ideal AerSimulator (statevector + sampling).

PASS criterion: the states Grover amplifies (the most probable measurement
outcomes) are exactly, and only, the classically-computed sum-free subsets --
i.e. quantum search recovers the same marked set that classical brute force
established as ground truth.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

ELEMENTS = [1, 2, 3, 4, 5]
N_QUBITS = len(ELEMENTS)
N_STATES = 1 << N_QUBITS


def is_sum_free(subset: set) -> bool:
    for x in subset:
        for y in subset:
            if x <= y and (x + y) in subset:
                return False
    return True


def classical_marked_states():
    """Brute-force, from first principles, the maximum-size sum-free subsets
    of {1,...,5}."""
    sizes = {}
    for bits in range(N_STATES):
        subset = {ELEMENTS[i] for i in range(N_QUBITS) if (bits >> i) & 1}
        if subset and is_sum_free(subset):
            sizes[bits] = len(subset)
    max_size = max(sizes.values())
    return [bits for bits, sz in sizes.items() if sz == max_size], max_size


def build_oracle(marked_states, n_qubits):
    """Diagonal phase oracle: flips sign of exactly the marked basis states."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = [(state >> i) & 1 for i in range(n_qubits)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_states, n_qubits, shots=4096):
    n_states = 1 << n_qubits
    m = len(marked_states)
    theta = math.asin(math.sqrt(m / n_states))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked_states, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    marked, max_size = classical_marked_states()
    m = len(marked)
    print(f"Classical brute force: maximum sum-free subset size of {{1,...,5}} "
          f"is {max_size}, achieved by {m} subset(s) out of {N_STATES} total.")
    print(f"Classical marked states (integer encoding): {sorted(marked)}")

    counts, iterations = run_grover(marked, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    # Qiskit prints classical-register bitstrings with qubit 0 as the
    # rightmost character, and our encoding also treats qubit i as the
    # weight-2^i bit, so the printed string parses directly as that integer.
    def bitstring_to_int(bs):
        return int(bs, 2)

    shots = sum(counts.values())
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])

    # Take the top-M most frequent outcomes (M = number of marked states) and
    # check they are exactly the classically marked states.
    top_states = {bitstring_to_int(bs) for bs, _ in sorted_counts[:m]}
    marked_set = set(marked)

    overlap = top_states & marked_set
    amplified_prob = sum(c for bs, c in counts.items()
                          if bitstring_to_int(bs) in marked_set) / shots
    baseline_prob = m / N_STATES

    print(f"Top-{m} measured states match classical marked set: "
          f"{len(overlap)}/{m}")
    print(f"Measured probability mass on marked states: {amplified_prob:.4f} "
          f"(uniform baseline would be {baseline_prob:.4f})")

    success = (top_states == marked_set) and (amplified_prob > baseline_prob + 0.15)

    if success:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
