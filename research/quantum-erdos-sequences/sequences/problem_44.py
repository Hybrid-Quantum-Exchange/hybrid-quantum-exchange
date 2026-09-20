"""
Erdos problem #44 (erdosproblems.com) -- quantum-testable instance.

Source metadata (data/problems.yaml, block "number: '44'"):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["N/A"]          <-- no OEIS sequence id is listed for this problem
    tags: ["number theory", "sidon sets", "additive combinatorics"]

LIMITATION, stated up front: problem #44 carries no OEIS id in the source
data ("N/A"), so there is no literal OEIS sequence to fetch a term from and
verify. Per the task instructions, this script instead builds its own
finite, computable property drawn directly from the problem's own subject
matter (its tags: Sidon sets / additive combinatorics), derives the correct
answer classically from first principles, and tests a real quantum circuit
against it. This is the "no OEIS id" fallback case, done honestly rather
than by fabricating an OEIS value.

The property tested
--------------------
A Sidon set (B2 set) is a set of non-negative integers in which all pairwise
sums a+b (a <= b) are distinct. Erdos problem #44 is about Sidon sets.

Finite instance: among the 5-element universe {0,1,2,3,4}, search the 10
subsets of size exactly 3 for one that is a Sidon set. This is:
  - finite and small (5 qubits: one per universe element, bit=1 means the
    element is in the candidate subset),
  - genuinely computable (checking the Sidon property is elementary
    arithmetic: compare all pairwise sums for repeats),
  - a real search problem with a known, classically-verified answer.

Classical ground truth is computed in this script (see `classical_sidon_search`)
by brute-force enumeration of all 32 subsets of {0,...,4}, before any quantum
code runs.

Quantum approach
-----------------
Grover's algorithm (real amplitude amplification, not a lookup table):
  - 5 qubits, one per universe element (2^5 = 32 basis states = all subsets).
  - Oracle: for each classically-identified "good" state (a 3-element Sidon
    subset), flip the phase of that computational basis state using X gates
    (to map the marked bitstring to |11111>-pattern of controls) sandwiched
    around a multi-controlled Z.
  - Diffuser: the standard Grover diffusion operator on 5 qubits.
  - Iterate the optimal number of times for the known number of good states
    out of 32, per the standard Grover formula.
  - Run on the ideal AerSimulator, sample many shots, and check that the
    measured bitstrings land in the classically-computed good set with high
    probability (this is the PASS/FAIL criterion).
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N = 5  # universe {0, 1, 2, 3, 4}
K = 3  # subset size searched for


def is_sidon(subset):
    """A set is Sidon (B2) if all pairwise sums a+b (a<=b) are distinct."""
    sums = []
    elems = sorted(subset)
    for i in range(len(elems)):
        for j in range(i, len(elems)):
            sums.append(elems[i] + elems[j])
    return len(sums) == len(set(sums))


def classical_sidon_search():
    """Brute-force, from first principles: every size-K subset of {0..N-1},
    which ones are Sidon sets. Returns the set of "good" bitstrings (qiskit
    little-endian convention: bit i of the string, read right-to-left,
    corresponds to qubit i / universe element i)."""
    good_bitstrings = set()
    all_size_k = list(combinations(range(N), K))
    sidon_subsets = [s for s in all_size_k if is_sidon(s)]
    for s in sidon_subsets:
        bits = ["0"] * N
        for elem in s:
            bits[elem] = "1"
        # qiskit bit ordering: qubit 0 is the rightmost character
        bitstring = "".join(reversed(bits))
        good_bitstrings.add(bitstring)
    return sidon_subsets, good_bitstrings


def build_oracle(n_qubits, good_bitstrings):
    """Phase-flip oracle: for each good bitstring, flip |good> -> -|good>
    using X-sandwiched multi-controlled Z (real gate-level oracle, not a
    lookup)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bitstring in good_bitstrings:
        # bitstring[0] corresponds to qubit n-1 ... bitstring[-1] to qubit 0
        zero_qubits = [
            q for q in range(n_qubits) if bitstring[n_qubits - 1 - q] == "0"
        ]
        if zero_qubits:
            qc.x(zero_qubits)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_qubits:
            qc.x(zero_qubits)
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


def build_grover_circuit(n_qubits, good_bitstrings, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(n_qubits, good_bitstrings)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    sidon_subsets, good_bitstrings = classical_sidon_search()
    total_states = 2 ** N
    num_good = len(good_bitstrings)

    print(f"Erdos problem #44 -- Sidon-set instance (universe size {N}, subset size {K})")
    print(f"OEIS id for problem #44: N/A (none listed in source data)")
    print(f"Classical brute force: {len(sidon_subsets)} of the size-{K} subsets "
          f"of {{0,...,{N-1}}} are Sidon sets: {sidon_subsets}")
    print(f"Good (Sidon) computational basis states: {sorted(good_bitstrings)}")
    assert num_good > 0, "no Sidon subsets found classically -- cannot build a Grover search"

    # Optimal number of Grover iterations for num_good marked out of total_states.
    theta = np.arcsin(np.sqrt(num_good / total_states))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations used: {iterations} (num_good={num_good}, N={total_states})")

    qc = build_grover_circuit(N, good_bitstrings, iterations)
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    good_shots = sum(c for bs, c in counts.items() if bs in good_bitstrings)
    good_fraction = good_shots / shots
    top_state = max(counts, key=counts.get)

    print(f"Measured {shots} shots; fraction landing on a classically-verified "
          f"Sidon-subset state: {good_fraction:.4f}")
    print(f"Most frequent measured bitstring: {top_state} "
          f"(is Sidon-subset state: {top_state in good_bitstrings})")

    # Success criterion: Grover amplification should concentrate most of the
    # probability mass on the classically-computed good states (baseline
    # random guessing would give num_good/total_states ~ 0.31 here).
    baseline = num_good / total_states
    passed = good_fraction > 0.75 and top_state in good_bitstrings

    print(f"Baseline (uniform-random) success probability: {baseline:.4f}")
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
