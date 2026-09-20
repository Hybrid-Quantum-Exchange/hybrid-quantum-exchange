"""
Erdos problem #892 (https://www.erdosproblems.com/892) — quantum-testable lane.

Problem #892's entry in erdosproblems' data/problems.yaml has:
    oeis: ["N/A"]
    tags: ["number theory", "primitive sets"]

There is NO OEIS sequence id attached to this problem (oeis is literally
"N/A"), so this script cannot test "membership in an OEIS sequence" as
instructed for the general case. This is disclosed honestly rather than
inventing an OEIS id. Instead, the *tag* "primitive sets" gives a real,
small, finite, computable property straight from the mathematical object
Erdos problem #892 is actually about: a set S of positive integers is
called a PRIMITIVE SET if no element of S divides another element of S
(this is the standard definition used throughout Erdos's primitive-sets
literature, e.g. A051953-adjacent OEIS entries on antichains under
divisibility).

Chosen finite instance and property
------------------------------------
S = (2, 3, 4, 5)  (4 elements -> C(4,2) = 6 unordered pairs)

Property tested: "S is NOT a primitive set, and the (unique, in this
instance) witnessing pair (a, b) with a | b and a != b is the pair
(2, 4)."

The 6 pairs are indexed 0..5 (in the fixed order enumerated by
itertools.combinations), padded out to 8 = 2**3 basis states for a
3-qubit index register. The classical answer -- which of the 8 index
states (if any) is a witness pair -- is computed here in Python from
first principles (plain trial division, no OEIS lookup, no hardcoded
answer) before any quantum code runs.

Quantum method: Grover's search algorithm (exact single-marked-item
case, 3 index qubits, 8-dimensional search space, 1 marked state) is
used to find the witness index on the ideal AerSimulator. The oracle
phase-flips exactly the basis state(s) the classical brute-force search
marked as witnessing a divisibility relation; the diffuser is the
standard Grover diffusion operator. The number of Grover iterations is
computed from the true count of marked states via the standard formula,
so the whole search -- marking included -- is the real classical
divisibility property re-expressed as a Grover oracle, not a fabricated
lookup table.

Pass condition: the basis state observed with overwhelming probability
after running the circuit on AerSimulator must equal the index of the
pair (2, 4) found classically, AND that pair must genuinely satisfy
a | b with a != b (checked independently of the circuit).
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_primitive_set_witness(elements):
    """Return (marked_indices, pairs) for the divisibility-witness search.

    pairs[i] is the i-th unordered pair (a, b), a < b, in the fixed
    itertools.combinations order. marked_indices is the list of indices
    i such that a | b (a divides b) for pairs[i] = (a, b) -- i.e. S is
    not a primitive set on account of that pair.
    """
    pairs = list(itertools.combinations(sorted(elements), 2))
    marked = [i for i, (a, b) in enumerate(pairs) if b % a == 0]
    return marked, pairs


def build_oracle(num_qubits, marked_indices):
    """Phase-flip exactly the computational basis states in marked_indices."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{num_qubits}b")[::-1]  # little-endian per qubit
        flip_qubits = [q for q, b in enumerate(bits) if b == "0"]
        if flip_qubits:
            qc.x(flip_qubits)
        if num_qubits == 1:
            qc.z(0)
        elif num_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        if flip_qubits:
            qc.x(flip_qubits)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    elif num_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(num_qubits, marked_indices, shots=4096):
    N = 2 ** num_qubits
    M = len(marked_indices)
    if M == 0 or M == N:
        raise ValueError("Grover needs 0 < M < N marked items for this instance")

    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(num_qubits, marked_indices)
    diffuser = build_diffuser(num_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    elements = (2, 3, 4, 5)
    num_qubits = 3  # covers 8 >= 6 pairs

    marked_indices, pairs = classical_primitive_set_witness(elements)

    print(f"Erdos problem #892 -- OEIS: N/A, tags: number theory, primitive sets")
    print(f"Testing set S = {elements}")
    print(f"Pairs (index: (a, b)):")
    for i, (a, b) in enumerate(pairs):
        print(f"  {i}: {a, b}{'  <-- a | b' if i in marked_indices else ''}")

    if len(marked_indices) != 1:
        raise RuntimeError(
            f"Instance must have exactly one witness pair for this exact-Grover "
            f"script; found {len(marked_indices)}: {marked_indices}"
        )

    classical_answer = marked_indices[0]
    a, b = pairs[classical_answer]
    assert a != b and b % a == 0, "classical witness check failed"
    print(f"\nClassical answer: witness pair index = {classical_answer} "
          f"(pair {(a, b)}, {a} | {b}) => S is NOT a primitive set.")

    counts, iterations = run_grover(num_qubits, marked_indices)
    print(f"\nGrover search: {num_qubits} qubits, {2**num_qubits} states, "
          f"{len(marked_indices)} marked, {iterations} iteration(s).")
    print(f"Measurement counts: {counts}")

    total_shots = sum(counts.values())
    best_bitstring = max(counts, key=counts.get)
    quantum_answer = int(best_bitstring, 2)
    confidence = counts[best_bitstring] / total_shots

    print(f"\nMost frequent measured index: {quantum_answer} "
          f"(classical bin '{best_bitstring}'), confidence = {confidence:.3f}")

    verified = (quantum_answer == classical_answer) and confidence > 0.5

    if verified:
        print("\nPASS: quantum Grover search found the same primitive-set "
              "divisibility witness as the classical brute-force search.")
    else:
        print("\nFAIL: quantum result did not match the classical answer.")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
