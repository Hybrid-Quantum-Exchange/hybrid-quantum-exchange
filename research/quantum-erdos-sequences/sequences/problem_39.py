"""
Erdos problem #39 (erdosproblems.com), quantum-testable instance.

Source metadata (data/problems.yaml, block "number: '39'"):
    prize: $500
    status: open
    tags: ["number theory", "sidon sets", "additive combinatorics"]
    oeis: ["N/A"]

LIMITATION: this problem has no associated OEIS sequence id in the source
data (oeis: ["N/A"]), so there is no OEIS-term property to target. Erdos
problem #39 is about Sidon sets (sets of integers with all pairwise sums
distinct), so instead of fabricating an OEIS lookup, this script targets a
small, finite, genuinely-computable property drawn straight from the
problem's own subject matter: "is a given finite set of integers a Sidon
set?", i.e. do all its pairwise sums a_i + a_j (i <= j) collide anywhere.

Classical instance chosen here (computed from first principles below, not
copied from any table):
    S = {0, 1, 3, 4}  (a small candidate set, deliberately NOT a Sidon set)

There are C(4,2) = 6 unordered pairs {i,j}, i<j, of indices into S:
    (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
These are enumerated as basis states 0..5 of a 3-qubit register (states
6,7 are unused/padding). For each pair we classically compute the sum
S[i]+S[j], and mark (as Grover targets) every pair whose sum equals the sum
of some *other* distinct pair -- i.e. every witness pair to S failing the
Sidon property. Grover's algorithm is run to amplify exactly those marked
basis states; the circuit's job is to *find the colliding pairs* by quantum
search, which we then check against the classical brute-force answer.

Circuit: standard 3-qubit Grover search (oracle built as a phase oracle via
multi-controlled Z gates on the classically-determined marked computational
basis states, diffusion = standard "inversion about the mean"), simulated
exactly on Qiskit Aer's statevector-based AerSimulator (no noise).

PASS criterion: after running Grover with the correct number of iterations
for this marked-count/space-size, the basis states measured with high
probability are exactly the classical set of colliding-pair indices (i.e.
quantum search found precisely the pairs that make S fail to be a Sidon
set).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_setup():
    """Build S, enumerate index pairs, compute sums, find collisions."""
    S = [0, 1, 3, 4]
    pairs = list(itertools.combinations(range(len(S)), 2))  # 6 pairs, i<j
    sums = [S[i] + S[j] for (i, j) in pairs]

    # A pair is a "Sidon violation witness" if its sum equals the sum of at
    # least one other (distinct) pair.
    marked_indices = []
    for idx, s in enumerate(sums):
        if sums.count(s) > 1:
            marked_indices.append(idx)

    is_sidon = len(marked_indices) == 0
    return S, pairs, sums, marked_indices, is_sidon


def build_grover_circuit(n_qubits, marked_indices):
    """3-qubit Grover search marking the given basis-state indices."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition.
    qc.h(range(n_qubits))

    N = 2 ** n_qubits
    num_marked = len(marked_indices)
    # Optimal number of Grover iterations for this N and marked count.
    theta = math.asin(math.sqrt(num_marked / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5)) if theta > 0 else 0

    def apply_oracle(circ):
        for m in marked_indices:
            bits = format(m, f"0{n_qubits}b")
            # Flip qubits that are 0 in this marked state so a
            # multi-controlled Z targets exactly |m>.
            for q, b in enumerate(reversed(bits)):
                if b == "0":
                    circ.x(q)
            if n_qubits == 1:
                circ.z(0)
            elif n_qubits == 2:
                circ.cz(0, 1)
            else:
                circ.h(n_qubits - 1)
                circ.mcx(list(range(n_qubits - 1)), n_qubits - 1)
                circ.h(n_qubits - 1)
            for q, b in enumerate(reversed(bits)):
                if b == "0":
                    circ.x(q)

    def apply_diffuser(circ):
        circ.h(range(n_qubits))
        circ.x(range(n_qubits))
        circ.h(n_qubits - 1)
        circ.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        circ.h(n_qubits - 1)
        circ.x(range(n_qubits))
        circ.h(range(n_qubits))

    for _ in range(iterations):
        apply_oracle(qc)
        apply_diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


def main():
    S, pairs, sums, marked_indices, is_sidon = classical_setup()

    print(f"S = {S}")
    print(f"pairs (index into S, i<j) = {pairs}")
    print(f"pairwise sums             = {sums}")
    print(f"classical marked (colliding-sum) pair indices = {marked_indices}")
    print(f"classical: is S a Sidon set? {is_sidon}")

    n_qubits = 3  # 8 basis states, covers 6 real pairs + 2 padding states
    qc, iterations = build_grover_circuit(n_qubits, marked_indices)
    print(f"Grover iterations used: {iterations}")

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Basis state index k corresponds to bitstring reversed (Qiskit little-endian).
    def bits_to_index(bitstring):
        # Qiskit's classical bitstring is written c[n-1]...c[0] (MSB first),
        # with c0 = qubit 0 = the LSB of the basis-state index -- i.e. it is
        # already ordinary big-endian binary for the index encoding used in
        # build_grover_circuit (which flips qubit q for the q-th bit from
        # the LSB). No reversal needed.
        return int(bitstring, 2)

    # Aggregate measured probability per basis-state index.
    probs = {}
    for bitstring, c in counts.items():
        idx = bits_to_index(bitstring)
        probs[idx] = probs.get(idx, 0) + c / shots

    # Determine which indices Grover amplified: any basis state whose
    # measured probability clearly exceeds the flat/no-marking baseline
    # 1/8 = 0.125 by a wide margin.
    threshold = 0.15
    found_indices = sorted(idx for idx, p in probs.items() if p > threshold)

    print(f"measured probability per basis index: {dict(sorted(probs.items()))}")
    print(f"quantum-found amplified indices: {found_indices}")

    verified = found_indices == sorted(marked_indices)

    print(f"classical marked indices:        {sorted(marked_indices)}")
    print(f"quantum amplified indices match: {verified}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
