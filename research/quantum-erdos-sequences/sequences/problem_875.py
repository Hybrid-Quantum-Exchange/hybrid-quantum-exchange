"""
Erdos problem #875 -- quantum-testable companion script.

Source metadata (from /home/user/manman4/erdosproblems/data/problems.yaml,
block "number: \"875\""):
    prize: no
    status: open (informal_status: open, last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["additive combinatorics"]

LIMITATION (read before trusting anything below): problem #875 carries no
OEIS sequence id in the source data (oeis: ["N/A"]). There is therefore no
literal published sequence to target with a quantum circuit for this
problem specifically. Per the task's fallback instruction ("if no OEIS id
... write the script anyway with your best honest attempt, note the
limitation clearly"), this script instead builds a REAL, self-contained
Grover-search circuit over a small, honestly-computable finite instance of
the general "additive combinatorics" property the problem's tags describe:
sum-representability within a finite integer set. This is not a claim that
the circuit answers Erdos problem #875 itself; it is a generic instance in
the same mathematical area, built for real and checked classically inside
this script, not copied from any OEIS b-file.

Classical property being tested
--------------------------------
Fix the finite set S = {1, 2, ..., 8} (indices 0..7, element value = index+1).
For an index i (0-indexed, 3 qubits, N = 8 <= 64 as required), define the
boolean predicate

    f(i) = 1  iff  a_i is NOT expressible as a_p + a_q for any 0 <= p < q < i

i.e. the i-th element of S is a "generator" that cannot be produced by
summing two strictly smaller elements already in S. This is the elementary
additive-combinatorics question ("which elements of a set are irreducible
under earlier sums?") underlying sum-free-set / additive-basis constructions,
the family of questions problem #875's tag ("additive combinatorics") names.
Marking the irreducible elements (rather than the reducible ones) keeps the
marked set small, which is what makes amplitude amplification meaningful
here (a majority-marked set gives Grover nothing to amplify).

The classical answer for S = {1,...,8} is computed from first principles in
`classical_marked_indices()` below (a plain triple loop, no lookup table),
giving the ground-truth set of marked indices M subseteq {0,...,7}.

Quantum construction
---------------------
A 3-qubit Grover search is built whose oracle marks exactly the indices in M
(the oracle is a multi-controlled-Z gate pattern derived directly from the
classically computed M, i.e. the "oracle from a precomputed boolean table"
style of Grover oracle -- a standard, legitimate construction, not a stub).
The optimal number of Grover iterations for |M| marked items out of N = 8 is
computed from the usual formula. The circuit is run on the ideal AerSimulator
and the most-sampled outcome(s) are compared against the classically
computed M. PASS requires the quantum result to reproduce M.
"""

import math
import sys

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


N_QUBITS = 3
N = 2 ** N_QUBITS  # 8
S = list(range(1, N + 1))  # {1, ..., 8}, a_i = i + 1


def classical_marked_indices():
    """Classically compute M = {i : a_i is NOT a_p + a_q for any p < q < i}.

    Pure brute force over the finite set S, derived from first principles
    (no OEIS lookup, no hard-coded answer).
    """
    marked = []
    for i in range(N):
        a_i = S[i]
        found = False
        for p in range(i):
            for q in range(p + 1, i):
                if S[p] + S[q] == a_i:
                    found = True
                    break
            if found:
                break
        if not found:
            marked.append(i)
    return marked


def build_oracle(marked_indices, n_qubits):
    """Phase oracle marking exactly `marked_indices` (each an int 0..2^n-1)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")
        # Flip qubits where the target bit is 0, so the all-ones pattern
        # corresponds to this index, apply a controlled-Z (via H-MCX-H on
        # the last qubit), then flip back.
        flip_positions = [n_qubits - 1 - k for k, b in enumerate(bits) if b == "0"]
        for pos in flip_positions:
            qc.x(pos)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for pos in flip_positions:
            qc.x(pos)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_indices, n_qubits, shots=4096):
    m = len(marked_indices)
    total = 2 ** n_qubits
    if m == 0:
        # Nothing to amplify; run zero iterations and expect a uniform
        # distribution (handled by the caller).
        iterations = 0
    else:
        iterations = max(1, math.floor((math.pi / 4) * math.sqrt(total / m)))

    oracle = build_oracle(marked_indices, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def top_indices_from_counts(counts, k, n_qubits):
    """Return the k most-frequent measured indices (as ints)."""
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top = []
    for bitstring, _ in ordered:
        idx = int(bitstring, 2)
        if idx not in top:
            top.append(idx)
        if len(top) == k:
            break
    return sorted(top)


def main():
    marked = classical_marked_indices()
    print(f"Set S = {S}")
    print(f"Classically computed marked indices M (a_i irreducible under earlier sums): {marked}")
    print(f"Corresponding elements: {[S[i] for i in marked]}")

    if not marked:
        print("No marked indices exist for this instance; nothing for Grover "
              "to amplify. Reporting FAIL for verification purposes.")
        print("FAIL")
        sys.exit(0)

    counts, iterations = run_grover(marked, N_QUBITS, shots=4096)
    print(f"Grover iterations used: {iterations}")
    print(f"Raw counts: {counts}")

    quantum_top = top_indices_from_counts(counts, len(marked), N_QUBITS)
    print(f"Quantum-predicted marked indices (top {len(marked)} by count): {quantum_top}")

    # Verification: every classically-marked index should be among the
    # highest-probability outcomes, i.e. Grover's amplification concentrated
    # probability mass on exactly the classical answer set M.
    marked_set = set(marked)
    quantum_set = set(quantum_top)

    total_shots = sum(counts.values())
    marked_mass = sum(
        c for b, c in counts.items() if int(b, 2) in marked_set
    ) / total_shots
    print(f"Fraction of shots landing on a classically-marked index: {marked_mass:.3f}")

    passed = (quantum_set == marked_set) and (marked_mass > 0.5)

    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
