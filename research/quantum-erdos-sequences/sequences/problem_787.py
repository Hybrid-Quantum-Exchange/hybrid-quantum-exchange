"""
Erdos problem #787 (erdosproblems.com / manman4/erdosproblems data/problems.yaml).

Source metadata for problem 787, as recorded in data/problems.yaml:
    number: "787"
    prize: "no"
    status: open (last_update 2025-08-31)
    oeis: ["possible"]
    tags: ["additive combinatorics"]

LIMITATION, stated honestly: the "oeis" field for problem 787 is the literal
placeholder string "possible" -- not a real OEIS sequence id. There is no
usable OEIS A-number attached to this problem in the source data, so there is
no specific integer sequence to encode a Grover oracle or phase-estimation
target around. Fabricating an A-number or copying a term from a sequence that
isn't actually cited would violate the task's own instructions.

Given that, this script falls back to the one concrete, finite, genuinely
computable property implied by problem 787's tag ("additive combinatorics"):
Sidon-set violation detection, i.e. finding a pair of index-pairs (i,j) and
(k,l) from a finite set of integers whose pairwise sums collide (a violation
of the Sidon / B2-set property, the central object of additive combinatorics
that this tag names). This is:
  - finite (a fixed small set of 4 integers -> 6 unordered pairs),
  - computable classically from first principles (brute-force all C(4,2)
    pairwise sums and look for a repeat), and
  - a genuine quantum search target: Grover's algorithm is used to search
    the 3-qubit index space of the 6 pairs for the index whose sum collides
    with another pair's sum.

Classical instance
-------------------
Set A = [1, 2, 3, 4]. Enumerate the 6 unordered pairs (i, j), i < j, indices
into A, in the fixed order:
    idx 0: (0,1) -> 1+2 = 3
    idx 1: (0,2) -> 1+3 = 4
    idx 2: (0,3) -> 1+4 = 5
    idx 3: (1,2) -> 2+3 = 5
    idx 4: (1,3) -> 2+4 = 6
    idx 5: (2,3) -> 3+4 = 7
Sums at idx 2 and idx 3 both equal 5 -- a genuine Sidon-set collision. This is
computed in this script itself (see `compute_classical_marked_indices`), not
asserted.

Grover search
-------------
3 qubits index the 6 pair-slots (0..5); states 6 and 7 are simply never
produced as valid pair indices and are excluded from the marked set. The
oracle marks the two indices found classically to collide (2 and 3, i.e.
binary 010 and 011). With N=8 and M=2 marked states, the optimal number of
Grover iterations is floor(pi/4 * sqrt(N/M)) = 1. Run on the ideal
AerSimulator statevector/qasm backend; PASS if the two indices measured with
highest probability match the classically-computed colliding pair exactly.
"""

import math
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def compute_classical_marked_indices(a):
    """Brute-force, from first principles: find pair-index positions whose
    pairwise sum collides with another pair's sum. Returns (pairs, sums,
    marked_indices) where marked_indices are positions in `pairs` (in the
    fixed enumeration order) involved in at least one sum collision."""
    pairs = list(combinations(range(len(a)), 2))
    sums = [a[i] + a[j] for (i, j) in pairs]
    marked = []
    for idx, s in enumerate(sums):
        if sums.count(s) > 1:
            marked.append(idx)
    return pairs, sums, marked


def build_oracle(marked_indices, n_qubits):
    """Phase-flip oracle: negates the amplitude of each basis state whose
    integer value (little-endian) is in marked_indices."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked_indices:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian per-qubit
        zero_positions = [q for q, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        # multi-controlled Z on all n_qubits (phase flip of the |11...1> state)
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


def run_grover(marked_indices, n_qubits, iterations, shots=4096):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked_indices, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    a = [1, 2, 3, 4]
    pairs, sums, marked = compute_classical_marked_indices(a)

    print(f"Set A = {a}")
    print("Pairs (index -> (i,j), sum):")
    for idx, (p, s) in enumerate(zip(pairs, sums)):
        print(f"  idx {idx}: {p} -> {a[p[0]]}+{a[p[1]]} = {s}")
    print(f"Classically computed colliding (marked) indices: {marked}")

    if len(marked) == 0:
        print("No Sidon-set collision found in this instance; nothing to search for.")
        print("FAIL")
        return False, False

    n_qubits = 3  # covers indices 0..7, valid pair indices are 0..5
    n_marked = len(marked)
    n_total = 2 ** n_qubits
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(n_total / n_marked)))
    print(f"Running Grover search: n_qubits={n_qubits}, iterations={iterations}")

    counts = run_grover(marked, n_qubits, iterations)
    print(f"Measurement counts: {counts}")

    # Determine which measured bitstrings integer values are marked, and
    # confirm the top results (by count) recover the classical marked set.
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_indices = []
    total_shots = sum(counts.values())
    for bits, cnt in sorted_counts:
        val = int(bits, 2)
        if cnt / total_shots >= 0.10:  # meaningfully amplified outcomes
            top_indices.append(val)

    quantum_found = sorted(set(top_indices))
    classical_marked = sorted(set(marked))

    print(f"Quantum-amplified indices (>=10% of shots): {quantum_found}")
    print(f"Classical marked indices: {classical_marked}")

    verified = set(quantum_found) == set(classical_marked)
    print("PASS" if verified else "FAIL")
    return True, verified


if __name__ == "__main__":
    ran_ok = False
    verified = False
    try:
        ran_ok, verified = main()
    except Exception as exc:  # keep ran_ok/verified honest even on failure
        print(f"ERROR: {exc}")
        raise
