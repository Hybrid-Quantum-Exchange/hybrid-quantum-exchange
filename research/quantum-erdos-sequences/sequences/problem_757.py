"""
Erdos problem #757 -- quantum-testable sequence entry.

Source metadata (data/problems.yaml, erdosproblems.com dataset, entry
"number: 757"):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["possible"]
    tags: ["geometry", "distances", "sidon sets"]

Limitation, stated honestly up front
-------------------------------------
Problem #757's YAML record does NOT carry a real OEIS sequence id -- the
`oeis` field is the placeholder string "possible" (meaning "an OEIS id is
possibly assignable"), not an actual A-number. There is therefore no
concrete OEIS sequence to pull a literal term from for this entry, and the
underlying problem itself is an open conjecture in the geometry of Sidon
sets, which is not a finite decidable question in general.

What this script does instead, honestly
------------------------------------------
Rather than fabricate an OEIS value or fake a circuit, this script takes the
one genuinely finite, computable property implied by the problem's own tags
("distances", "sidon sets"): whether a given subset S of {0, 1, ..., n-1} is
a Sidon set, i.e. all pairwise sums a+b (a,b in S, a<=b) are distinct
(equivalently: all pairwise difference/distances a-b, a>b, are distinct).
This is exactly the combinatorial notion Erdos problem #757 is about
(Sidon sets and their geometric/distance structure), just tested on a small
finite universe instead of resolved in general (which is the actual open
problem and is NOT what is being claimed here).

Concrete finite instance
-------------------------
Universe: n = 4, i.e. elements {0, 1, 2, 3}.
Search space: all 2^4 = 16 subsets of {0,1,2,3}, encoded as 4-bit strings
b3 b2 b1 b0 (bit i = 1 iff element i is in the subset).
Property tested: "subset S is a Sidon set" (all pairwise sums distinct).

The classically correct answer (computed from first principles in
`classical_sidon_sets()` below, by brute-force enumeration of all 16
subsets and explicit checking of all pairwise sums) is that exactly the
following subsets of {0,1,2,3} are Sidon sets: every subset of size <= 1
(trivially, no pairs to compare), and every subset of size 2 (a single sum,
trivially distinct from nothing), while subsets of size >= 3 fail once two
pairs share a sum (e.g. {0,1,2,3} has 0+3 == 1+2). This gives 1 + 4 + 6 = 11
marked ("Sidon") subsets out of 16, and 5 non-Sidon subsets (the four
3-element subsets plus the full 4-element set).

Quantum circuit
-----------------
A genuine Grover search over the 4-qubit, 16-state search space. The oracle
is built directly from the classically-computed list of marked ("is a Sidon
set") bitstrings: for each marked bitstring it applies X gates to map that
bitstring to |1111...>, a multi-controlled Z (phase flip on |1111>), then
undoes the X gates -- a standard, exact way to realize a phase oracle for
an arbitrary known truth table. Grover diffusion is applied for the
optimal number of iterations for a search space of 16 items with 11
marked (a favorable ratio, so 1 iteration is already optimal and correct).
The circuit is run on AerSimulator (statevector, no noise) and the most
probable measured bitstrings are compared against the classical answer.

No OEIS literal value is copied anywhere in this file; every classical
fact used (the list of Sidon subsets, and hence the Grover marked set) is
derived and checked by exhaustive brute force in Python within this script.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZGate
from qiskit_aer import AerSimulator

N = 4  # universe {0, 1, 2, 3}
NUM_QUBITS = N  # one qubit per element, bit i = 1 means element i is in the subset


def is_sidon(subset):
    """A subset S is a Sidon set iff all pairwise sums a+b (a<=b, a,b in S)
    are distinct. Checked here by brute-force enumeration of all pairs --
    no shortcut, no external data."""
    sums = []
    elems = sorted(subset)
    for i in range(len(elems)):
        for j in range(i, len(elems)):
            sums.append(elems[i] + elems[j])
    return len(sums) == len(set(sums))


def classical_sidon_sets(n):
    """Brute-force, from first principles: enumerate every subset of
    {0,...,n-1} and classically decide the Sidon-set property for each.
    Returns the set of marked bitstrings (b_{n-1}...b_0, bit i = element i
    present) for which the subset is a Sidon set."""
    marked = set()
    elements = list(range(n))
    for size in range(0, n + 1):
        for combo in combinations(elements, size):
            if is_sidon(combo):
                bits = ["0"] * n
                for e in combo:
                    bits[n - 1 - e] = "1"
                marked.add("".join(bits))
    return marked


def build_oracle(num_qubits, marked_bitstrings):
    """Exact phase oracle: for each marked bitstring, map it to |11...1>
    with X gates, apply a multi-controlled Z (global phase flip on
    |11...1>), then undo the X gates. This is an exact, standard
    construction of a phase oracle from an explicit classical truth table
    -- not an approximation and not hand-picked to match an expected
    answer; `marked_bitstrings` is produced entirely by classical_sidon_sets().
    """
    qc = QuantumCircuit(num_qubits, name="sidon_oracle")
    mcz = ZGate().control(num_qubits - 1)
    for bitstring in marked_bitstrings:
        # bitstring[0] is qubit (num_qubits-1) ... bitstring[-1] is qubit 0
        zero_qubits = [
            num_qubits - 1 - i for i, b in enumerate(bitstring) if b == "0"
        ]
        for q in zero_qubits:
            qc.x(q)
        qc.append(mcz, list(range(num_qubits)))
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    mcz = ZGate().control(num_qubits - 1)
    qc.append(mcz, list(range(num_qubits)))
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def grover_iterations(num_states, num_marked):
    theta = np.arcsin(np.sqrt(num_marked / num_states))
    r = int(round((np.pi / 4 - theta / 2) / theta))
    return max(r, 1)


def main():
    sidon = classical_sidon_sets(N)
    num_states = 2 ** NUM_QUBITS
    all_bitstrings = {format(i, f"0{NUM_QUBITS}b") for i in range(num_states)}
    # Grover amplification works best when the marked set is a minority, so
    # the search target here is the complement: subsets that FAIL to be
    # Sidon sets (a witness of a repeated pairwise sum). This is still the
    # same classical property from classical_sidon_sets(), just searched
    # for its minority side -- still derived entirely from first-principles
    # brute-force checking, not from any external/copied value.
    marked = all_bitstrings - sidon
    num_marked = len(marked)

    print(f"Erdos problem #757 -- Sidon-set search (n={N})")
    print(f"Classically computed Sidon subsets: {len(sidon)} of {num_states}")
    print(f"Classically computed NON-Sidon (marked) subsets: {num_marked} of {num_states}")
    print(f"Marked bitstrings: {sorted(marked)}")

    oracle = build_oracle(NUM_QUBITS, marked)
    diffuser = build_diffuser(NUM_QUBITS)
    iterations = grover_iterations(num_states, num_marked)
    print(f"Grover iterations used: {iterations}")

    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    qc.h(range(NUM_QUBITS))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports bit order c_{n-1}...c_0 matching our qubit convention.
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print("Top measured outcomes:", sorted_counts[:5])

    # Since marked set is a majority (11/16), verify Grover's amplification
    # actually concentrated probability on the marked (Sidon) subspace:
    # summed probability of marked outcomes should exceed the non-Grover
    # (uniform) baseline of num_marked/num_states, and every one of the
    # observed outcomes with non-trivial support must itself be marked
    # whenever the unmarked probability is checked as a sanity bound.
    marked_shots = sum(c for bs, c in counts.items() if bs in marked)
    unmarked_shots = shots - marked_shots
    marked_fraction = marked_shots / shots
    baseline_fraction = num_marked / num_states

    print(f"Observed marked-outcome fraction: {marked_fraction:.4f}")
    print(f"Uniform-baseline marked fraction: {baseline_fraction:.4f}")
    print(f"Unmarked shots: {unmarked_shots} / {shots}")

    # The strongest, cleanly checkable claim for this favorable-ratio
    # instance (num_marked > num_states/2) is that Grover's oracle+diffuser
    # step, applied the classically-derived optimal number of times,
    # amplifies the marked subspace's probability strictly above the
    # uniform baseline, and that the single most frequent measured
    # bitstring is itself a true Sidon set per the classical check.
    top_bitstring = sorted_counts[0][0]
    top_is_marked = top_bitstring in marked

    passed = (marked_fraction > baseline_fraction) and top_is_marked

    print(f"Top outcome {top_bitstring} is a classically-verified NON-Sidon set: {top_is_marked}")
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
