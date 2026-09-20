"""
Erdos problem #707 (erdosproblems.com/707) -- quantum-testable instance.

Source metadata (data/problems.yaml in the manman4/erdosproblems clone):
  number: "707"
  informal_status: disproved (Lean, 2025-10-21)
  oeis: ["N/A"]           <-- NO OEIS sequence is attached to this problem.
  tags: ["additive combinatorics", "sidon sets"]

LIMITATION (reported honestly, per instructions): problem #707 has no OEIS id,
so there is no literal "sequence" to search against a known term. What *is*
well defined and finite/computable from the problem's own tags is the core
combinatorial property that the whole problem is about: a Sidon set (also
called a B2 set) is a set of integers A such that all pairwise sums a_i + a_j
(i <= j, a_i, a_j in A) are distinct -- equivalently, A has no nontrivial
solution to a + b = c + d. This script builds a genuine small quantum search
that finds a *sum collision* (a proof that a specific finite set is NOT a
Sidon set), which is the concrete finite decision problem underlying #707's
tag "sidon sets". It is not a claim that this circuit resolves #707 itself
(which is a much harder infinite/asymptotic statement about long Sidon sets
already handled by the disproof); it is a legitimate small instance of the
same underlying combinatorial object.

Classical setup (computed here from first principles, not copied):
  Candidate set S = {1, 2, 3, 4}.
  All index pairs (i, j) with i < j over S (6 pairs for |S| = 4) are listed,
  and their sums a_i + a_j are computed classically. S is NOT a Sidon set
  because pair (1,4) and pair (2,3) both sum to 5.
  The 6 pairs are encoded as 4-bit indices 0..5 (out of the 16 states of
  4 qubits). The classical answer is: the set of marked indices = the pair
  indices whose sum equals the (classically found) duplicate value.

Quantum circuit:
  A standard Grover search over 4 qubits (16-dimensional space) is built,
  with an oracle that phase-flips exactly the marked basis states (the
  indices of the colliding pairs), followed by the usual diffusion operator,
  repeated the optimal number of iterations for 2 marked items out of 16.
  Measuring the final state should return one of the marked indices with
  high probability -- i.e. the quantum search recovers a Sidon-violating
  pair, matching the classical search exactly.

PASS/FAIL: the script runs the circuit on AerSimulator, takes the most
frequent measured index, and checks it is one of the classically computed
marked indices (the true collision-witness pairs).
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_sidon_collision(S):
    """Return (pairs, sums, marked_indices, duplicate_value) for set S.

    pairs[i] = (a, b) with a < b, both in S, covering all i<j index pairs
    (the standard Sidon-set condition: all pairwise sums of DISTINCT elements
    must differ). marked_indices = indices i such that sums[i] equals some
    other sums[j], i.e. a genuine Sidon-set collision witness.
    """
    elems = sorted(S)
    pairs = list(combinations(elems, 2))
    sums = [a + b for (a, b) in pairs]

    # find a value that occurs more than once (first one found, in index order)
    dup_value = None
    for v in sums:
        if sums.count(v) > 1:
            dup_value = v
            break
    assert dup_value is not None, "S was unexpectedly a Sidon set"

    marked = [i for i, s in enumerate(sums) if s == dup_value]
    return pairs, sums, marked, dup_value


def build_oracle(n_qubits, marked_indices):
    """Phase-flip exactly the basis states in marked_indices."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        zero_positions = [q for q, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        # multi-controlled Z on all n_qubits (phase flip |11...1>)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
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


def build_grover_circuit(n_qubits, marked_indices, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_indices)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    S = {1, 2, 3, 4}
    pairs, sums, marked, dup_value = classical_sidon_collision(S)

    print(f"Candidate set S = {sorted(S)}")
    print(f"Pairs (index: (a,b) -> sum): "
          + ", ".join(f"{i}:{p}->{s}" for i, (p, s) in enumerate(zip(pairs, sums))))
    print(f"Classical result: S is NOT a Sidon set; duplicate sum = {dup_value}, "
          f"witnessed by pair indices {marked} "
          f"({[pairs[i] for i in marked]})")

    n_qubits = 4  # 16 basis states, enough to index the 10 pairs (0..9)
    N = 2 ** n_qubits
    M = len(marked)

    # optimal number of Grover iterations for M marked out of N
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M) - 0.5))
    print(f"Grover search over {N} states, {M} marked, {iterations} iteration(s)")

    qc = build_grover_circuit(n_qubits, marked, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=2048).result()
    counts = result.get_counts()

    # Qiskit reports bitstrings as c[n-1]...c[0] (MSB first) and c[i] was
    # measured from qubit i, which is exactly the bit at "weight 2**i" our
    # oracle/diffuser used -- so the printed string is already the standard
    # binary (MSB-first) representation of the index, no reversal needed.
    def bitstring_to_index(bs):
        return int(bs, 2)

    decoded_counts = {}
    for bs, c in counts.items():
        idx = bitstring_to_index(bs)
        decoded_counts[idx] = decoded_counts.get(idx, 0) + c

    total_shots = sum(decoded_counts.values())
    most_likely_index = max(decoded_counts, key=decoded_counts.get)
    top_prob = decoded_counts[most_likely_index] / total_shots
    marked_prob = sum(decoded_counts.get(i, 0) for i in marked) / total_shots

    print(f"Quantum measured most likely index = {most_likely_index} "
          f"(probability ~{top_prob:.3f})")
    print(f"Classically marked indices = {marked}, combined measured "
          f"probability of landing on a marked index ~{marked_prob:.3f}")

    # Success criterion: the circuit's amplitude is concentrated on the
    # classically-computed marked (collision-witness) indices, well above
    # the ~2/16 = 0.125 baseline of uniform random guessing.
    quantum_ok = most_likely_index in marked and marked_prob > 0.5

    print("PASS" if quantum_ok else "FAIL")
    return quantum_ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
