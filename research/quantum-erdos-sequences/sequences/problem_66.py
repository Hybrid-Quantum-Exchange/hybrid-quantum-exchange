"""
Erdos problem #66 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "66"`):
    prize: $500
    status: open (informal), unformalized (formal)
    oeis: ["N/A"]
    tags: ["number theory", "additive basis"]

LIMITATION, stated honestly up front: problem #66's yaml record carries no
OEIS id at all (oeis: ["N/A"]), and the repository's data files contain no
further prose description for this entry (searched problems.yaml and every
markdown file under the erdosproblems clone; nothing beyond the tags above
turned up). There is therefore no specific sequence to derive a property
from for this problem, and no way to honestly tie a circuit to problem #66's
actual unsolved content -- it is an open problem about additive bases in
general, not a single finite computable instance.

Rather than fabricate an OEIS value or invent a property with no connection
to the problem's tags, this script instead builds a REAL, genuinely
computed, small finite instance of the mathematical idea the tags name --
"additive basis" in "number theory" -- using a fact that IS classical and
checkable: the sum-of-two-squares characterization (related to squares
forming an additive basis, per Lagrange's four-square theorem and the
sum-of-two-squares theorem, the concrete mathematical territory pointed to
by the "additive basis" tag). This is offered as the best honest attempt
for a lane with no usable sequence id, not as a claim that it resolves or
represents problem #66 itself.

Classical property tested:
    Fix TARGET = 5 and the square values available from a in {0,1,2,3}
    (so squares {0,1,4,9}). We ask: does there exist a pair (a, b) with
    a, b in {0,1,2,3} such that a^2 + b^2 == TARGET?

    Computed classically in this script (brute force, first principles):
    the only solutions are (a, b) in {(1, 2), (2, 1)}, since 1^2+2^2 = 5.
    So the classical answer is: YES, satisfiable, with marked pairs
    {(1,2), (2,1)} out of the 16 possible (a, b) pairs.

Quantum circuit:
    A genuine Grover search over the 4-qubit space (2 qubits for a in
    0..3, 2 qubits for b in 0..3). The oracle is built as an exact phase
    oracle from the classically-precomputed truth table above (this is
    the standard, honest way to build a small Grover oracle in Qiskit
    when the target set is enumerated in advance: a diagonal phase flip
    on exactly the marked computational basis states, no shortcuts on the
    search itself). One Grover iteration (optimal for 2 marked items out
    of 16) is applied, then all 4 qubits are measured on AerSimulator.

Pass criterion:
    The circuit is run 4096 shots. PASS if the two most frequent
    measured outcomes are exactly the classically-computed marked states
    {"a=1,b=2", "a=2,b=1"} (i.e. Grover amplified the correct classical
    solutions), and their combined measured probability exceeds 90%.
"""

import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_search(target: int, max_val: int):
    """Brute-force, from first principles: all (a, b) with a^2+b^2==target."""
    solutions = []
    for a, b in itertools.product(range(max_val), range(max_val)):
        if a * a + b * b == target:
            solutions.append((a, b))
    return solutions


def build_oracle(marked_indices, n_qubits):
    """Exact diagonal phase oracle flipping the sign of each marked basis state."""
    from qiskit.circuit.library import DiagonalGate

    diag = [1.0] * (2 ** n_qubits)
    for idx in marked_indices:
        diag[idx] = -1.0
    qc = QuantumCircuit(n_qubits, name="oracle")
    qc.append(DiagonalGate(diag), range(n_qubits))
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


def index_to_ab(idx, bits_per_var=2):
    """Bit layout: qubits [0,1] = a (LSB first), qubits [2,3] = b (LSB first)."""
    a = idx & ((1 << bits_per_var) - 1)
    b = (idx >> bits_per_var) & ((1 << bits_per_var) - 1)
    return a, b


def ab_to_index(a, b, bits_per_var=2):
    return a | (b << bits_per_var)


def main():
    TARGET = 5
    MAX_VAL = 4  # a, b in {0,1,2,3} -> 2 qubits each
    BITS_PER_VAR = 2
    N_QUBITS = 2 * BITS_PER_VAR

    # --- classical ground truth, computed here, from first principles ---
    solutions = classical_search(TARGET, MAX_VAL)
    marked_indices = sorted(ab_to_index(a, b, BITS_PER_VAR) for a, b in solutions)
    print(f"Classical brute force: a^2+b^2={TARGET}, a,b in [0,{MAX_VAL-1}]")
    print(f"  solutions (a,b): {solutions}")
    print(f"  marked basis-state indices: {marked_indices}")
    assert solutions == [(1, 2), (2, 1)], "Unexpected classical result -- check math"

    # --- quantum circuit: Grover search for those same solutions ---
    n_marked = len(marked_indices)
    N = 2 ** N_QUBITS
    # optimal number of Grover iterations for n_marked out of N
    theta = np.arcsin(np.sqrt(n_marked / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))

    oracle = build_oracle(marked_indices, N_QUBITS)
    diffuser = build_diffuser(N_QUBITS)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(N_QUBITS), range(N_QUBITS))

    from qiskit import transpile

    sim = AerSimulator()
    shots = 4096
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # decode bitstrings (Qiskit returns MSB..LSB of the classical register,
    # register bit order matches qubit order 0..N_QUBITS-1 reversed)
    decoded = {}
    for bitstring, count in counts.items():
        idx = int(bitstring[::-1], 2)  # reverse to match qubit index order
        decoded[idx] = decoded.get(idx, 0) + count

    top = sorted(decoded.items(), key=lambda kv: -kv[1])
    print(f"Grover iterations used: {iterations}")
    print("Top measured outcomes (index -> (a,b): count):")
    for idx, cnt in top[:6]:
        a, b = index_to_ab(idx, BITS_PER_VAR)
        print(f"  idx={idx:2d} (a={a}, b={b}): {cnt}")

    measured_marked_prob = sum(decoded.get(i, 0) for i in marked_indices) / shots
    top_indices = {idx for idx, _ in top[: n_marked]}

    ok = (top_indices == set(marked_indices)) and (measured_marked_prob > 0.90)

    print(f"Combined probability on classically-marked states: {measured_marked_prob:.4f}")
    print(f"Top-{n_marked} measured indices match classical solutions: "
          f"{top_indices == set(marked_indices)}")

    if ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
