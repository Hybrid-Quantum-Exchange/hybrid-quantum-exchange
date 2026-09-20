"""
Erdos problem #863 -- quantum-testable instance.

Source metadata (erdosproblems/data/problems.yaml, entry "number: \"863\"",
tags ["number theory", "sidon sets", "additive combinatorics"]):
    oeis: ["N/A"]
This problem has NO associated OEIS sequence id -- the metadata field is
literally "N/A". Per the task instructions ("if no genuine quantum circuit
can be constructed ... write the script anyway with your best honest
attempt, note the limitation clearly"), this script does not use any OEIS
sequence (there is none to use). Instead it builds a genuine, small,
finite, computable property drawn directly from the problem's own tags
(Sidon sets / additive combinatorics), since that is the closest real
mathematical content available for this entry.

Classical property tested
--------------------------
A Sidon set (B2 sequence) is a set of integers such that all pairwise sums
a_i + a_j (i <= j) are distinct. This script takes the small candidate set

    S = [0, 1, 3, 4]

and forms all C(4,2) = 6 unordered pairs of distinct elements, each
producing a sum. It computes classically, from first principles (no
lookup), which pair-indices collide with some other pair-index (i.e. share
an equal sum) -- these are exactly the pairs that certify S is *not* a
Sidon set. This is a small, finite, fully computable decision property.

For S = [0, 1, 3, 4] the 6 pairs (indexed 0..5) and sums are:
    0: (0,1) -> 1
    1: (0,3) -> 3
    2: (0,4) -> 4
    3: (1,3) -> 4   <-- collides with pair 2 (both sum to 4)
    4: (1,4) -> 5
    5: (3,4) -> 7
So the "colliding" (marked) pair-indices are exactly {2, 3}: this
witnesses that S is NOT a Sidon set (0+4 == 1+3).

Quantum circuit
----------------
A Grover search over the 3-qubit index space {0,...,7} (6 real pair-slots,
padded to a power of two) is built whose oracle marks exactly the
classically-determined colliding indices {2, 3} (binary 010, 011) via a
phase flip, followed by the standard Grover diffuser. With 2 marked out of
8 states, one Grover iteration is optimal. The circuit is run on the ideal
AerSimulator and the most probable measured basis states are compared
against the classical marked set {2, 3}.

PASS criterion: the two most frequently measured 3-bit strings (interpreted
as integers) equal the classical marked-index set {2, 3} exactly.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_colliding_pair_indices(S):
    """Return the set of pair-indices whose sum collides with another pair's sum,
    computed from first principles (brute-force over all pairs)."""
    n = len(S)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, S[i] + S[j]))

    sums = [p[2] for p in pairs]
    marked = set()
    for idx, s in enumerate(sums):
        for jdx, s2 in enumerate(sums):
            if idx != jdx and s == s2:
                marked.add(idx)
    return marked, pairs


def build_oracle(num_qubits, marked_ints):
    """Phase-flip oracle marking the given integers (as num_qubits-bit strings)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for m in marked_ints:
        bits = format(m, f"0{num_qubits}b")
        # flip qubits that should be 0 so the marked pattern becomes all-1s
        for q, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        elif num_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for q, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(q)
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


def run_grover(num_qubits, marked_ints, iterations, shots=4096):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(num_qubits, marked_ints)
    diffuser = build_diffuser(num_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    S = [0, 1, 3, 4]
    marked, pairs = classical_colliding_pair_indices(S)
    print(f"Candidate set S = {S}")
    print("Pairs and sums:")
    for idx, (i, j, s) in enumerate(pairs):
        flag = "  <-- collides" if idx in marked else ""
        print(f"  {idx}: S[{i}]+S[{j}] = {S[i]}+{S[j]} = {s}{flag}")
    print(f"Classical marked (colliding) pair-indices: {sorted(marked)}")
    print("=> S is NOT a Sidon set (classically verified).")

    num_qubits = 3  # indices 0..7, 6 real pair-slots padded to 8
    num_marked = len(marked)
    # Optimal Grover iteration count for N=8, M marked items.
    N = 2 ** num_qubits
    theta = np.arcsin(np.sqrt(num_marked / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    counts = run_grover(num_qubits, marked, iterations, shots=4096)
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print(f"\nGrover ran with {iterations} iteration(s), top measured outcomes:")
    for bitstring, c in sorted_counts[:5]:
        print(f"  {bitstring} (index {int(bitstring, 2)}): {c} shots")

    top_k = sorted_counts[: num_marked]
    top_indices = set(int(bs, 2) for bs, _ in top_k)

    verified = top_indices == marked
    print(f"\nQuantum top-{num_marked} measured indices: {sorted(top_indices)}")
    print(f"Classical marked indices:               {sorted(marked)}")

    if verified:
        print("\nPASS")
    else:
        print("\nFAIL")


if __name__ == "__main__":
    main()
