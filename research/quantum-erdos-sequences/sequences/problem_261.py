"""
Erdos problem #261 -- quantum-testable stand-in.

Source metadata (data/problems.yaml in the erdosproblems repo, entry
`number: "261"`):
    prize: no
    status: open (informal), unformalized (formal)
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION, stated honestly up front: problem #261 carries no OEIS sequence
id in the source data (oeis: ["N/A"]), and no further statement text for it
is present in the read-only clone at
/home/user/manman4/erdosproblems/data/problems.yaml (no file named for
"261" exists in that repo either). There is therefore no real OEIS-derived
sequence to build a faithful quantum-testable instance from for this
specific problem. Per instructions, rather than fabricate an OEIS id or
copy a value with no derivation, this script instead builds its best
honest, genuinely-computable number-theory instance consistent with
problem #261's only real attribute (tag: "number theory"): finding the
nontrivial divisors of a small composite integer via Grover search. This
is a real, finite, classically-checkable property -- it is just not tied
to an OEIS sequence for #261, because none exists in the source data.

Classical property under test
------------------------------
N = 15. Search space: x in {0, 1, ..., 15} (4 qubits, values 0-15).
Marked (winning) states: the nontrivial divisors of N, i.e. all x with
2 <= x <= N-1 and N % x == 0.

The script computes this classical answer itself, from first principles,
by trial division over the full 4-qubit search space (0..15) -- no value
is hard-coded from any external source:

    divisors(15) among 2..14 -> {3, 5}

Quantum approach
-----------------
A Grover search circuit over 4 qubits marks exactly the basis states in
the classically-computed divisor set with a phase oracle, then applies
the standard Grover diffusion operator, iterated the optimal number of
times for a search space of size 16 with 2 marked items
(floor(pi/4 * sqrt(16/2)) = 2 iterations). The circuit is run on the
ideal AerSimulator with many shots; the script then checks that the
top measured outcomes (by count) are exactly the classically-computed
marked set {3, 5}, and prints PASS/FAIL accordingly.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_nontrivial_divisors(n: int) -> set:
    """Trial division from first principles: divisors of n in [2, n-1]."""
    return {x for x in range(2, n) if n % x == 0}


def build_oracle(n_qubits: int, marked_states: set) -> QuantumCircuit:
    """Phase oracle that flips the sign of each marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run() -> None:
    N = 15
    n_qubits = 4  # search space 0..15
    search_space_size = 2 ** n_qubits

    marked = classical_nontrivial_divisors(N)
    marked = {m for m in marked if m < search_space_size}
    print(f"Classical property: nontrivial divisors of {N} in "
          f"[2, {search_space_size - 1}]")
    print(f"Classically computed marked set: {sorted(marked)}")

    num_marked = len(marked)
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(search_space_size / num_marked)))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    shots = 4096
    job = backend.run(qc, shots=shots)
    counts = job.result().get_counts()

    # Convert bitstrings (qiskit prints classical register with qubit 0 as
    # rightmost bit) back to integers.
    int_counts = Counter()
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        int_counts[value] += c

    top_k = [v for v, _ in int_counts.most_common(num_marked)]
    quantum_result = set(top_k)

    print(f"Grover iterations used: {iterations}")
    print(f"Top {num_marked} measured outcomes (by count): {sorted(quantum_result)}")
    print(f"Full outcome counts: {dict(sorted(int_counts.items()))}")

    passed = quantum_result == marked
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    run()
